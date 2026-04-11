#!/bin/bash
cd /root/yanyijin/STdiff/models
python train_from_yaml.py --config config/lwrdiff_96_12_v2.yaml > logs/train_96_12_v2.log 2>&1
