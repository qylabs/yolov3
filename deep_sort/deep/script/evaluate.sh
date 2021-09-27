#!/bin/bash
# features_file=checkpoint/features.pth #for market1501
# features_file=checkpoint/features_cust.pth
# features_file=checkpoint/features_cust_reverse.pth


# features_file=checkpoint/features_resnet18.pth
# features_file=checkpoint/features_resnet18_cust.pth
features_file=checkpoint/features_resnet18_cust_reverse.pth

python evaluate.py ${features_file}