# VQ-GAN (3D) + MS-SPADE Block

## Overview

![VQGAN](/asset/VQGAN.jpg)

We transform the source latent into a target-like latent style through the Multi Switchable SPADE block.



## Requirements

A suitable [conda](https://conda.io/) environment named `vqgan` can be created and activated with:

```
conda env create -f environment.yaml
conda activate vqgan
```

If you encounter environmental conflicts or version errors, please try the following methods.

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```



## Data Preparation

#### BraTS 2021

You can download the BraTS2021 dataset from the following [link](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863 ). It should have the following structure:

```
$/data/BraTS2021/TrainingData
├── BraTS2021_00000
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── ...
├── BraTS2021_00002
│   ├── BraTS2021_00002_t1.nii.gz
│   ├── BraTS2021_00002_t1ce.nii.gz
│   ├── ...
├── ...
```



## Training

Training can be started by running

```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py -b configs/brats_vqgan.yaml -t True --gpus 0,
```



## Inference

```
python scripts/samples.py -b configs/brats_vqgan.yaml --outdir=/outdir -r vqgan.ckpt
```





## Thanks 
Thanks to everyone who contributed code and models.

- [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers) 
- [Semantic Image Synthesis with SPADE](https://github.com/NVlabs/SPADE)

