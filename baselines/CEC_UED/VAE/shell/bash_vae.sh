#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python baselines/CEC_UED/VAE/train_vae.py \
    NUM_EPOCHS=100 \
    TUNE=True \
    LAYOUT_DATA_FILE=layout_dataset_1e6_all.h5 \
    PROJECT=crossenv_ued_vae_sweep

CUDA_VISIBLE_DEVICES=0 python baselines/CEC_UED/VAE/train_vae.py \
    NUM_EPOCHS=100 \
    TUNE=True \
    LAYOUT_DATA_FILE=layout_dataset_1e7_all.h5 \
    PROJECT=crossenv_ued_vae_sweep