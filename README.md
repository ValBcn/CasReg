# CasReg

## Introduction




CasReg is a deep learning model based on cascaded networks, that produce small amounts of displacement to warp progressively the moving image towards the fixed image. 

The trained registration model can then be used to perform Multi-Atlas Segmentation (MAS) : multiple annotated images and their labels are registered with the image to segment, the resulting warped labels are then combined to form a refined segmentation.

<p align="center">
  <img src="https://github.com/ValBcn/CasReg/blob/master/images/casc_net.png?raw=true" title="Overview of the cascaded registration method" width=70% height=70%>
</p>

This repository includes:
  - A preprocessing script, that convert Nifti images (and optionally labels) to .npz format. The images are then cropped, resized and normalized.
  - Training and testing script for cascaded registration.
  - A multi-atlas segmentation script using the trained weights of the cascaded registration.

For more information about CasReg, please read the following paper:

**Unsupervised fetal brain MR segmentation using multi-atlas deep learning registration**\
Valentin Comte<sup>1</sup>, Mireia Alenya<sup>1</sup>, Andrea Urru<sup>1</sup>, Ayako Nakaki<sup>2</sup>, Francesca Crovetto<sup>2</sup>, Oscar Camara<sup>1</sup>, Elisenda Eixarch<sup>2</sup>, Fàtima Crispi<sup>2</sup>, Gemma Piella Fenoy<sup>1</sup>, Mario Ceresa<sup>1</sup>, and Miguel A. González Ballester<sup>1,3</sup>

<sup>1 BCN MedTech, Department of Information and Communication Technologies, Universitat Pompeu
Fabra, Barcelona, Spain\
2 Maternal Fetal Medicine, BCNatal, Center for Maternal Fetal and Neonatal Medicine (Hospital Clínic
and Hospital Sant Joan de Déu), Barcelona, Spain\
3 ICREA, Barcelona, Spain</sup>

## Installation

### Prerequisites

CasReg requires a GPU for training and inference (it should have at least 10GB of memory).

### Dependencies

CasReg uses th following packages:

- pytorch-gpu 1.9.0
- torchvision 0.2.2
- torchsummary 1.5.1
- cudatoolkit 10.2
- numpy 1.21
- nibabel 4.0.1
- simpleitk 2.1.1
- scipy 1.7.3

We highly recommend to use a Conda environment to install the package dependencies. To install all the required packages, run:

```
conda env create -f casreg_env.yml
```

## Preprocessing

To run the preprocessing, use the following command:

```
python preprocessing.py --img_path /path/to/nifti/images/folder/ --label_path /path/to/nifti/labels/folder/ --prep_dir /path/to/preprocessed/folder/ --img_size 128 128 128
```
img_size defines the new shape of the input images (and labels), default is (128,128,128).

## Training

To run the training, use the following command:

```
python train.py --save_dir /path/to/the/weights/folder/ --npz_dir /path/to/preprocessed/folder/ --nb_labels 8 --nb_cascades 5 --contracted 1
```

save_dir: the weights will be saved there./
nb_labels: is the number of labels of the segmentation used for validation (optional)./
img_size: defines the new shape of the input images (and labels), default is (128,128,128)./
contracted: 0 for original architecture (uses more memory) / 1 for the contracted architecture./

## Testing

## Multi-atlas segmentation
