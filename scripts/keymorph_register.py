import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from argparse import ArgumentParser
import torchio as tio
from pathlib import Path

from keymorph.utils import rescale_intensity, align_img
from keymorph.model import KeyMorph
from keymorph.unet3d.model import UNet2D, UNet3D, TruncatedUNet3D
from keymorph.net import ConvNet
# from pairwise_register_eval import run_eval
# from groupwise_register_eval import run_group_eval
from script_utils import summary, load_checkpoint

'''
Brainmorph register, works on full resolution.
'''

def parse_args():
    parser = ArgumentParser()

    # I/O
    parser.add_argument(
        "--gpus", type=str, default="0", help="Which GPUs to use? Index from 0"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        dest="save_dir",
        default="./register_output/",
        help="Path to the folder where outputs are saved",
    )
    parser.add_argument(
        "--load_path", type=str, default=None, help="Load checkpoint at .h5 path"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./weights/",
        help="Directory where keymorph model weights are saved",
    )
    parser.add_argument(
        "--save_eval_to_disk", action="store_true", help="Perform evaluation"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize images and points"
    )
    parser.add_argument("--debug_mode", action="store_true", help="Debug mode")

    # KeyMorph
    parser.add_argument(
        "--registration_model", type=str, default="keymorph", help="Registration model"
    )
    parser.add_argument(
        "--num_keypoints", type=int, required=True, help="Number of keypoints"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="truncatedunet",
        help="Keypoint extractor module to use",
    )
    parser.add_argument(
        "--num_truncated_layers_for_truncatedunet",
        type=int,
        default=1,
        help="Number of truncated layers for truncated unet",
    )
    parser.add_argument(
        "--num_levels_for_unet",
        type=int,
        default=4,
        help="Number of levels for unet",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="instance",
        choices=["none", "instance", "batch", "group"],
        help="Normalization type",
    )

    parser.add_argument(
        "--weighted_kp_align",
        type=str,
        default="power",
        choices=[None, "variance", "power"],
        help="Type of weighting to use for keypoints",
    )

    parser.add_argument(
        "--list_of_aligns",
        nargs="*",
        default=("affine",),
        help="Alignments to use for KeyMorph",
    )

    parser.add_argument(
        "--list_of_metrics",
        nargs="*",
        default=("mse",),
        help="Metrics to report",
    )

    # Data
    parser.add_argument("--moving", type=str, required=True, help="Moving image path")

    parser.add_argument("--fixed", type=str, required=True, help="Fixed image path")

    parser.add_argument("--moving_seg", type=str, default=None, help="Moving seg path")

    parser.add_argument("--fixed_seg", type=str, default=None, help="Fixed seg path")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--moved", type=str, required=True, help="Moved save path")
    
    # parser.add_argument("--trans", type=str, required=True, help="Transformation save path")
    
    parser.add_argument(
        "--half_resolution",
        action="store_true",
        help="Evaluate on half-resolution models",
    )

    parser.add_argument(
        "--early_stop_eval_subjects",
        type=int,
        default=None,
        help="Early stop number of test subjects for fast eval",
    )

    # JR: Add real world coords flag
    parser.add_argument(
        "--align_keypoints_in_real_world_coords",
        action="store_true",
        help="Align keypoints in real world coords",
    )
    
    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=23,
        help="Random seed use to sort the training data",
    )

    parser.add_argument("--dim", type=int, default=3)

    parser.add_argument("--use_amp", action="store_true", help="Use AMP")

    parser.add_argument(
        "--groupwise", action="store_true", help="Perform groupwise registration"
    )

    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Use torch.utils.checkpoint",
    )

    parser.add_argument(
        "--num_resolutions_for_itkelastix", type=int, default=4, help="Num resolutions"
    )

    args = parser.parse_args()
    return args


def build_tio_subject(img_path, seg_path=None):
    _dict = {"img": tio.ScalarImage(img_path)}
    if seg_path is not None:
        _dict["seg"] = tio.LabelMap(seg_path)
    return tio.Subject(**_dict)


def get_loaders(args):
    if os.path.isfile(args.moving) and os.path.isfile(args.fixed):
        moving = [build_tio_subject(args.moving)]
        fixed = [build_tio_subject(args.fixed)]

    # Build dataset
    fixed_dataset = tio.SubjectsDataset(fixed, transform=transform)
    moving_dataset = tio.SubjectsDataset(moving, transform=transform)
    fixed_loader = DataLoader(
        fixed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    moving_loader = DataLoader(
        moving_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    loaders = {"fixed": fixed_loader, "moving": moving_loader}
    return loaders

def load_img(img_path, norm = True):
    ''' Load image and normalize as is done in get_loader.
    Our images are already volumes of 256**3 isotropic 1-mm voxels, so
    resampling, cropping, and padding are unnecessary.'''
    f = tio.ScalarImage(img_path)
    if norm:
        f = tio.Lambda(rescale_intensity)(f)
        
    # Transpose and flip image such that its axes are roughly aligned with RAS.
    ori_to_world = f.affine
    f = tio.ToCanonical()(f)
    ras_to_world = f.affine
    ori_to_ras = np.linalg.inv(ras_to_world) @ ori_to_world

    return ori_to_ras, ras_to_world, f.data.unsqueeze(0).float()

def load_seg(seg_path):
    ''' Load seg as is done in get_loader.
    Our images are already volumes of 256**3 isotropic 1-mm voxels, so
    resampling, cropping, and padding are unnecessary.'''
    f = tio.LabelMap(seg_path)

    # Transpose and flip image such that its axes are roughly aligned with RAS.
    ori_to_world = f.affine
    f = tio.ToCanonical()(f)
    ras_to_world = f.affine
    ori_to_ras = np.linalg.inv(ras_to_world) @ ori_to_world

    return ori_to_ras, ras_to_world, f.data.unsqueeze(0).float()

def get_foundation_weights_path(weights_dir, num_keypoints, num_levels):
    template_name = "foundation-numkey{}-numlevels{}.pth.tar"
    return os.path.join(weights_dir, template_name.format(num_keypoints, num_levels))


def get_model(args):
    if args.registration_model == "keymorph":
        # CNN, i.e. keypoint extractor
        if args.backbone == "conv":
            network = ConvNet(
                args.dim,
                1,
                args.num_keypoints,
                norm_type=args.norm_type,
            )
        elif args.backbone == "unet":
            if args.dim == 2:
                network = UNet2D(
                    1,
                    args.num_keypoints,
                    final_sigmoid=False,
                    f_maps=64,
                    layer_order="gcr",
                    num_groups=8,
                    num_levels=args.num_levels_for_unet,
                    is_segmentation=False,
                    conv_padding=1,
                )
            if args.dim == 3:
                network = UNet3D(
                    1,
                    args.num_keypoints,
                    final_sigmoid=False,
                    f_maps=32,  # Used by nnUNet
                    layer_order="gcr",
                    num_groups=8,
                    num_levels=args.num_levels_for_unet,
                    is_segmentation=False,
                    conv_padding=1,
                    use_checkpoint=args.use_checkpoint,
                )
        elif args.backbone == "truncatedunet":
            if args.dim == 3:
                network = TruncatedUNet3D(
                    1,
                    args.num_keypoints,
                    args.num_truncated_layers_for_truncatedunet,
                    final_sigmoid=False,
                    f_maps=32,  # Used by nnUNet
                    layer_order="gcr",
                    num_groups=8,
                    num_levels=args.num_levels_for_unet,
                    is_segmentation=False,
                    conv_padding=1,
                )
        else:
            raise ValueError('Invalid keypoint extractor "{}"'.format(args.backbone))
        network = torch.nn.DataParallel(network)

        # Keypoint model
        registration_model = KeyMorph(
            network,
            args.num_keypoints,
            args.dim,
            use_amp=args.use_amp,
            use_checkpoint=args.use_checkpoint,
            weight_keypoints=args.weighted_kp_align,
            # align_keypoints_in_real_world_coords=args.align_keypoints_in_real_world_coords, 
        )
        registration_model.to(args.device)
        summary(registration_model)
    elif args.registration_model == "itkelastix":
        from keymorph.baselines.itkelastix import ITKElastix

        registration_model = ITKElastix()
    elif args.registration_model == "synthmorph":

        from keymorph.baselines.voxelmorph import VoxelMorph

        registration_model = VoxelMorph(perform_preaffine_register=True)
    elif args.registration_model == "synthmorph-no-preaffine":

        from keymorph.baselines.voxelmorph import VoxelMorph

        registration_model = VoxelMorph(perform_preaffine_register=False)
    elif args.registration_model == "ants":
        from keymorph.baselines.ants import ANTs

        registration_model = ANTs()
    else:
        raise ValueError(
            'Invalid registration model "{}"'.format(args.registration_model)
        )
    return registration_model


if __name__ == "__main__":
    args = parse_args()

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpus))
    else:
        args.device = torch.device("cpu")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("Number of GPUs: {}".format(torch.cuda.device_count()))
    print(f"Torch version is {torch.__version__}")

    # Create save path
    save_path = Path(args.save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.model_eval_dir = save_path

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model
    registration_model = get_model(args)
    registration_model.eval()

    # Checkpoint loading
    if args.half_resolution and args.registration_model == "keymorph":
        assert (
            args.load_path is not None
        ), "Must specify path for weights for half resolution models"
    else:
        args.load_path = get_foundation_weights_path(
            args.weights_dir, args.num_keypoints, args.num_levels_for_unet
        )
    if args.load_path is not None:
        print(f"Loading checkpoint from {args.load_path}")
        ckpt_state, registration_model = load_checkpoint(
            args.load_path,
            registration_model,
            device=args.device,
        )
    
    # Data
    fix_to_vox, aff_f, img_f = load_img(args.fixed)
    mov_to_vox, aff_m, img_m = load_img(args.moving)
    # Move to device
    img_f = img_f.float().to(args.device)
    img_m = img_m.float().to(args.device)
    # aff_f = torch.from_numpy(np.expand_dims(aff_f, axis=0))
    # aff_m = torch.from_numpy(np.expand_dims(aff_m, axis=0))
    
    # Register
    with torch.set_grad_enabled(False):
        registration_results = registration_model(
            img_f,
            img_m,
            transform_type=args.list_of_aligns,
            return_aligned_points=True,
            # align_keypoints_in_real_world_coords = args.align_keypoints_in_real_world_coords,
            # aff_f=aff_f.to(img_f),
            # aff_m=aff_m.to(img_m),
        )

    # Get matrix
    save_mappings = {'affine': 'aff',
                     'rigid' : 'rig'}
    # Still didnt find a way to extract transform...
    # for align_type_str, res_dict in registration_results.items():
    #     transform = res_dict["matrix"][0].cpu().detach().numpy()
    #     transform = np.linalg.inv(mov_to_vox) @ np.linalg.inv(transform) @ fix_to_vox # TODO: fix this, not correct
        
    #     # Save matrix.
    #     save_name = args.trans
    #     directory, filename = os.path.split(save_name)
    #     modified_filename = filename.replace("aff", save_mappings[align_type_str])
    #     save_name = os.path.join(directory, modified_filename)
    #     print(f'Saving matrix as {save_name}') 
    #     np.savetxt(fname=save_name, X=transform, fmt='%.8f %.8f %.8f %.8f')

    # Get moved
    if args.moved:
        # Resampling: original intensities (no normalization).
        *_, img_m = load_img(args.moving, norm=False)
        *_, seg_m = load_seg(args.moving_seg)
        for align_type_str, res_dict in registration_results.items():
            print(f'Processing alignment type: {align_type_str}')
            grid = res_dict["grid"].cpu().detach()
            img_a = align_img(grid, img_m, mode = 'bilinear')
            seg_a = align_img(grid, seg_m, mode = 'nearest')
            
            # Save image.
            save_name = args.moved
            directory, filename = os.path.split(save_name)
            modified_filename = filename.replace("aff", save_mappings[align_type_str])
            save_name = os.path.join(directory, modified_filename)
            print(f'Saving moved image as {save_name}') 
            out = tio.ScalarImage(tensor=img_a[0, ...].cpu(), affine=aff_f)
            out.save(path=save_name, squeeze=True)

            # Save seg.
            modified_filename = modified_filename.replace("ima", "lab")
            save_name = os.path.join(directory, modified_filename)
            print(f'Saving moved seg as {save_name}') 
            out = tio.LabelMap(tensor=seg_a[0, ...].cpu(), affine=aff_f)
            out.save(path=save_name, squeeze=True)


