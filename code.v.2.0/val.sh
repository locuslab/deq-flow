#!/bin/bash

python -u main.py --eval --name deq-flow-H-all-grad --stage things \
    --validation kitti sintel --restore_ckpt checkpoints/deq-flow-H-things-test-1x.pth --gpus 0 \
    --wnorm --f_thres 40 --f_solver naive_solver  \
    --eval_factor 1.5 --huge

python -u main.py --eval --name deq-flow-H-all-grad --stage things \
    --validation kitti sintel --restore_ckpt checkpoints/deq-flow-H-things-test-3x.pth --gpus 0 \
    --wnorm --f_thres 40 --f_solver naive_solver  \
    --eval_factor 1.5 --huge

python -u main.py --eval --name deq-flow-H-all-grad --stage things \
    --validation kitti sintel --restore_ckpt checkpoints/deq-flow-H-things-test-3x.pth --gpus 0 \
    --wnorm --f_thres 40 --f_solver naive_solver  \
    --eval_factor 3.0 --huge


