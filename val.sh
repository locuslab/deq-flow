#!/bin/bash

python -u main.py --eval --name deq-flow-B-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-B-1-step-grad-things.pth --gpus 0 \
    --wnorm --f_thres 40 --f_solver anderson 

python -u main.py --eval --name deq-flow-B-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-B-things.pth --gpus 0 \
    --wnorm --f_thres 60 --f_solver anderson 

python -u main.py --eval --name deq-flow-H-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-things-1.pth --gpus 0 \
    --wnorm --f_thres 36 --f_solver broyden --huge 

python -u main.py --eval --name deq-flow-H-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-things-2.pth --gpus 0 \
    --wnorm --f_thres 36 --f_solver broyden --huge 

python -u main.py --eval --name deq-flow-H-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-things-3.pth --gpus 0 \
    --wnorm --f_thres 36 --f_solver broyden --huge 
