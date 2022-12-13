# CasReg

## Introduction

![network](images/casc_net_small.png)

CasReg is a deep learning model based on cascaded networks, that produce small amounts of displacement to warp progressively the moving image towards the fixed image. Once the networks are trained, multiple annotated magnetic resonance (MR) fetal brain images and their labels are registered with the image to segment, the resulting warped labels are then combined to form a refined segmentation.
