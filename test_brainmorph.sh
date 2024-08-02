#!/bin/sh
'''
Test BrainMorph demo: full-resolution models.
Using one of my data.
'''
python scripts/register.py \
    --num_keypoints 256 \
    --num_levels_for_unet 6 \
    --weights_dir ./weights/ \
    --moving /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/brainmask_m00.nii.gz \
    --fixed /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/brainmask_m24.nii.gz \
    --moving_seg /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/synthseg_brainmask_m00.nii.gz \
    --fixed_seg  /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/synthseg_brainmask_m24.nii.gz \
    --list_of_aligns rigid affine \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --visualize