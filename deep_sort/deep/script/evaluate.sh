#!/bin/bash
features_file=checkpoint/features.pth
# features_file=checkpoint/features_cust.pth
# features_file=checkpoint/features_cust_reverse.pth

python evaluate.py ${features_file}