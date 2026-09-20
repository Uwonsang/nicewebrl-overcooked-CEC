#! /bin/bash
gpu=$0

# IPPO
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py --config-name=test_general_toy model_name=IPPO 

# FCP
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py --config-name=test_general_toy model_name=FCP

# CEC
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py --config-name=test_general_toy model_name=CEC

