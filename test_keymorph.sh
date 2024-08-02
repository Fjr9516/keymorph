#!/bin/sh
'''
Test Keymorph demo: IXI-trained, half-resolution models
Uring one of my data.
'''
python scripts/register.py \
    --half_resolution \
    --num_keypoints 128 \
    --backbone conv \
    --moving /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/brainmask_m00.nii.gz \
    --fixed /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/brainmask_m24.nii.gz \
    --load_path ./weights/numkey128_aff_dice.1560.h5 \
    --moving_seg /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/synthseg_brainmask_m00.nii.gz \
    --fixed_seg  /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/synthseg_brainmask_m24.nii.gz \
    --list_of_aligns rigid affine \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --visualize