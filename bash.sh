#!/usr/bin/env bash

#python main.py --ckpt pretrained_models/metfaces.pt --PCA_path pretrained_models/PCA_metfaces.npz --size 1024

#python main.py --ckpt pretrained_models/afhqcat.pt --PCA_path pretrained_models/PCA_afhqcat.npz --size 512
#
#python main.py --ckpt pretrained_models/afhqwild.pt --PCA_path pretrained_models/PCA_afhqwild.npz --size 512

python main.py --ckpt pretrained_models/afhqdog.pt --PCA_path pretrained_models/PCA_afhqdog.npz --size 512