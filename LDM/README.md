# Latent Diffusion Models (3D)

## Overview

![VQGAN](/asset/LDM.jpg)



## Requirements

A suitable [conda](https://conda.io/) environment named `vqgan` can be created and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

If you encounter environmental conflicts or version errors, please try the following methods.

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install kornia==0.7.1 --no-deps
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
python main.py -b configs/latent-diffusion/brats-ldm-vq-4.yaml -t True --gpus 0,
```



## Inference

```
python scripts/sample.py  -b configs/latent-diffusion/brats-ldm-vq-4.yaml --ddim_eta 1.0
```



## Thanks 

Thanks to everyone who contributed code and models.

- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)

