#!/bin/bash
# features_file=checkpoint/features.pth #for market1501
# features_file=checkpoint/features_cust.pth
# features_file=checkpoint/features_cust_reverse.pth  #0.667


# features_file=checkpoint/features_resnet18.pth
# features_file=checkpoint/features_resnet18_cust.pth
# features_file=checkpoint/features_resnet18_cust_reverse.pth


# features_file=checkpoint/features_mobilenetv2_x1_0.pth
# features_file=checkpoint/features_mobilenetv2_x1_0_r256.pth
# features_file=checkpoint/features_mobilenetv2_x1_0_r128_cust_reverse.pth  #0.814
# features_file=checkpoint/features_mobilenetv2_x1_0_r256_cust_reverse.pth  #0.801

# features_file=checkpoint/features_osnet_x0_25_r256_cust_reverse.pth

python evaluate.py ${features_file}