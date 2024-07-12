#!/bin/sh
'''
Test Keymorph demo: IXI-trained, half-resolution models
'''
python scripts/register.py \
    --half_resolution \
    --num_keypoints 128 \
    --backbone conv \
    --moving ./example_data_half/img_m/IXI_001_128x128x128.nii.gz \
    --fixed ./example_data_half/img_m/IXI_002_128x128x128.nii.gz \
    --load_path ./weights/numkey128_aff_dice.1560.h5 \
    --moving_seg ./example_data_half/seg_m/IXI_001_128x128x128.nii.gz \
    --fixed_seg ./example_data_half/seg_m/IXI_002_128x128x128.nii.gz \
    --list_of_aligns affine \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --visualize