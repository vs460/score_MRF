#!/bin/bash


python main_fastmri.py \
 --config=configs/ve/MRF_config.py \
 --eval_folder=eval/fastmri_multicoil_knee_320 \
 --mode='train'  \
 --workdir=workdir/fastmri_multicoil_knee_320