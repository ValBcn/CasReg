# CasReg

## Introduction

<p align="center">
  <img src="https://github.com/ValBcn/CasReg/blob/master/images/casc_net.png?raw=true" title="Overview of the cascaded registration method" width=70% height=70%>
</p>


CasReg is a deep learning model based on cascaded networks, that produce small amounts of displacement to warp progressively the moving image towards the fixed image. 

The trained registration model can then be used to perform Multi-Atlas Segmentation (MAS) : multiple annotated images and their labels are registered with the image to segment, the resulting warped labels are then combined to form a refined segmentation.

This repository includes:
  - A preprocessing script, that convert Nifti images (and optionally labels) to .npz format. The images are then cropped, resized and normalized.
  - Training and testing script for cascaded registration.
  - A multi-atlas segmentation script using the trained weights of the cascaded registration.

For more information about CasReg, please read the following paper:

*Unsupervised fetal brain MR segmentation using multi-atlas deep learning registration*\
Valentin Comte<sup>1</sup>, Mireia Alenya1, Andrea Urru1, Ayako Nakaki2, Francesca Crovetto2, Oscar Camara1, Elisenda Eixarch2, Fàtima Crispi2, Gemma Piella Fenoy1, Mario Ceresa1, and Miguel A. González Ballester1,2.
