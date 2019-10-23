# Synthesizing 3D Shapes from Silhouette Image Collections using Multi-projection Generative Adversarial Networks

The main contributors of this repository include [Xiao Li](http://pableeto.github.io), [Yue Dong](http://yuedong.shading.me), [Pieter Peers](http://www.cs.wm.edu/~ppeers/) and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/).

## Introduction

This repository provides a reference implementation for the CVPR 2019 paper "Synthesizing 3D Shapes from Silhouette Image Collections using Multi-projection Generative Adversarial Networks".

More information (including a copy of the paper) can be found at https://pableeto.github.io/publications/projgan/.

## Citation
If you use our code or models, please consider cite:

```
@conference{Li:2019:S3S,
author = {Li, Xiao and Dong, Yue and Peers, Pieter and Tong, Xin},
title = {Synthesizing 3D Shapes from Silhouette Image Collections using Multi-Projection Generative Adversarial Networks},
month = {June},
year = {2019},
booktitle = {IEEE International Conference on Computer Vision and Pattern Recognition},
}
```

----------------------------------------------------------------
## Usage:

### System Requirements
   - Linux system (validated on Ubuntu 16.04 system)
   - At least one NVidia GPU (validated on Titan X and GTX 1080Ti)
   - Python 3.6 (Python 2.x is not supported)
   - Tensorflow 1.14 or above (<2.0)
   - Tensorflow graphics
   - Tensorflow probaility 0.7.0
   - Numpy
   - opencv-python
   - py-vox-io
   - tqdm

We have provided a requirements.txt file for quickly install all dependencies. You can install all packages with: 

    pip install -r requirements.txt

We also provide a docker file with all required libs installed, please check ./docker/dockerfile for details.

### Installation
After installing all the prerequisites listed above, download (or git clone) the code repository. 
Furthermore, to retrain the network, you may also need to download the datasets which contains silouette images of 3D objects (see below).

### Play with pre-trained models
We provide pre-trained model of our 3D shape generator on chair (32^3) and bird (64^3). They are trained with only 2D silouette image collections.
You can download the pretrained model here: XXX

To generate some voxel 3D shapes from pre-trained models, please run:

    python run_generation.py --model_ckpt $CHECKPOINTS$ --out_folder $OUT_FOLDER$ --category $CATEGORY$ --number_to_gen $N$ --gpuid $GPUID$

with:

**$CHECKPOINTS$:** path of model checkpoints.

**$OUT_FOLDER$:** output folder.

**$CATEGORY$:** model type (chair\_32\_syn | bird\_64\_real)

**$N$:** number of voxel to generate.

**$GPUID$:** GPU id to use.

The generated voxel is saved with .vox format. You can visualize it with MagicaVoxel (https://ephtracy.github.io/)

### Training from scratch
### Datasets
We provide training datasets (chair and bird) for re-training our model. The dataset can be downloaded from: https://drive.google.com/open?id=10A554EDdfIKVly9sY5A6M6Gw-dNzcvk1. 

The dataset are packed into TFRecord format and can be directly used for training our model.

Note: the bird silouettes are cropped from [1] and the chair silouettes are generated from [2].
Please kindly cite their paper if you use the corrsponded data.

After download and extract the data. You can run the training with:

    python train_voxel_vp_mp_gan.py --output_dir $OUTPUT_DIR$ --data_dir $DATA_DIR$ --cfg_file_gan ./experiment_setup/gan.yaml --cfg_file_classifier ./experiment_setup/classifier.yaml --gpu_id $GPUID$

with:

**$$OUTPUT_DIR$:** output folder for dumping checkpoints and intermediate results.

**$DATA_DIR$:** folder containing training data.

**$GPUID$:** GPU id to use.

The detailed training parameters for training GAN and classifier (e.g. batchsize, network structures, voxel and image resolutions, etc) can be modified in "cfg\_file\_gan" (i.e. ./experiment_setup/gan.yaml) and 
"cfg\_file\_classifier" (i.e. ./experiment_setup/classifier.yaml) respectively.

### Preparing your own data
Here are some instructions if you want to train our model with your own data:

-   The training data of our model should contains 2D silouette image of one certain category of objects. You may need to utilize some state-of-the-art segmentation methods to extract silouette mask for real images.
-   Our code take TFRecord format as training data input. We provide a python script for convert a folder of silouette images into TFRecord; please check ./scripts/img\_to\_tfrecord.py for details.

## Contact
Please contact Xiao Li (pableetoli@gmail.com) if you have any problems.

## Reference
[1] Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001.

[2] Chang, Angel X., et al. "Shapenet: An information-rich 3d model repository." arXiv preprint arXiv:1512.03012 (2015).