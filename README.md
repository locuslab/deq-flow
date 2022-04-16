# Deep Equilibrium Optical Flow Estimation

This is the official repo for paper, *Deep Equilibrium Optical Flow Estimation* (CVPR 2022).

<div align=center><img src="assets/frame_0037_frame.png" width="512" height="218" /></div>
<div align=center><img src="assets/frame0037_pred.png" width="512" height="218" /></div>

## Demo

https://user-images.githubusercontent.com/18630903/163676562-e14a433f-4c71-4994-8e3d-97b3c33d98ab.mp4

## Requirements

Install required environments through the following commands.

```bash
conda create --name deq python==3.6.10
conda activate deq
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install tensorboard scipy opencv matplotlib einops termcolor -c conda-forge
```

Download the datasets into the `datasets` directory.

[FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)

[FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

[MPI Sintel](http://sintel.is.tue.mpg.de/)

[KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

[HD1k](http://hci-benchmark.iwr.uni-heidelberg.de/)

## Inference

Download pretrained [checkpoints]() into the `checkpoints` directory. Run the following command to infer over the Sintel train set and the KITTI train set.

```bash
bash val.sh
```

You may expect the following performance statistics of given checkpoints. This is a reference [log](./log/val.log).

|  Checkpoint Name | Sintel (clean) | Sintel (final) | KITTI AEPE  | KITTI F1-all |
| :--------------: | :------------: | :------------: | :---------: | :----------: |
| DEQ-Flow-B   | 1.43 | 2.79 | 5.43 | 16.67 |
| DEQ-Flow-H-1 | 1.45 | 2.58 | 3.97 | 13.41 |
| DEQ-Flow-H-2 | 1.37 | 2.62 | 3.97 | 13.62 |
| DEQ-Flow-H-3 | 1.36 | 2.62 | 4.02 | 13.92 |

## Visualization

Download pretrained [checkpoints]() into the `checkpoints` directory. Run the following command to visualize the optical flow estimation over the KITTI test set.

```bash
bash viz.sh
```

## Training

Download Chairs pretrained [checkpoints]() into the `checkpoints` directory.

For the efficiency mode, you can run 1-step gradient to train DEQ-Flow-B via the following command. Memory overhead per GPU is about 5800 MB. This can be further reduced when combined with `--mixed-precision`. You may expect best results of about 1.46 (AEPE) on Sintel (clean), 2.85 (AEPE) on Sintel (final), 5.29 (AEPE) and 16.24 (F1-all) on KITTI. This is a reference [log](./log/B_1_step_grad.log).

```bash
bash train_B_demo.sh
```

For training a demo of DEQ-Flow-H, you can run this command.

```bash
bash train_H_demo.sh
```

To train DQE-Flow-B on Chairs and Things, use the following command.

```bash
bash train_B.sh
```

For the performance mode, you can run this command to train DEQ-Flow-H using the ``C+T`` and ``C+T+S+K+H`` schedule. You may expect the performance of < 1.40 (AEPE) on Sintel (clean), 2.60 (AEPE) on Sintel (final), around 4.00 (AEPE) and 13.6 (F1-all) on KITTI. DEQ-Flow-H-1,2,3 are checkpoints from three runs. The best results usually peak at about 80k to 90k on Things.

This training protocol needs three 11 GB GPUs. In the next several months, an upcoming implementation revision will further reduce this overhead to **less than two 11 GB GPUs**.

```bash
bash train_H_full.sh
```

## A Tutorial on DEQ

If you hope to learn more about DEQ, here is a [tutorial](https://implicit-layers-tutorial.org/) on implicit deep learning. Enjoy yourself!

## Reference

If you find our work helpful to your research, please consider citing this paper. :)

```bib
@inproceedings{deq-flow,
    author = {Bai, Shaojie and Geng, Zhengyang and Savani, Yash and Kolter, J. Zico},
    title = {Deep Equilibrium Optical Flow Estimation},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}
```

## Contact

Feel free to contact us if you have additional questions. Please drop me an email through zhengyanggeng@gmail.com. Find me at [Twitter](https://twitter.com/ZhengyangGeng).
