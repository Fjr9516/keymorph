#!/bin/sh
'''
Test BrainMorph register: full-resolution models. 
Better get transform (txt/lta), then moved ima and lab.
Using one of my data.
'''
python scripts/keymorph_register.py \
    --num_keypoints 256 \
    --num_levels_for_unet 6 \
    --weights_dir ./weights/ \
    --moving /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/brainmask_m00.nii.gz \
    --fixed /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/brainmask_m24.nii.gz \
    --moving_seg /autofs/space/bal_004/users/jf1212/code/affine_synthmorph/data/eval/adni-3t/0689/synthseg_brainmask_m00.nii.gz \
    --list_of_aligns rigid affine \
    --moved ./register_output/aff.ima.1.nii.gz \
    --save_dir ./register_output/