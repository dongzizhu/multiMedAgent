#!/bin/bash

python src/demo_mentors_final.py --use_VQA --VQA_dir data/data_all_with_diff --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/MMQ_BAN_MEVF_all --maml_nums 2,5 --model BAN --epoch _best --batch_size 32 --load --split train


