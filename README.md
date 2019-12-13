# Readers for medical imaging datasets

The goal of this repository is to help researchers and practitioners working with medical imaging datasets and reduce an amount of routine work.

In order to use the functions from this repository you should download a dataset that you need from [Grand Challenges in Biomedical Image Analysis](https://grand-challenge.org/challenges/).

The repository contains code for reading a dataset into memory and for auxiliary tasks: 
* resize images by bilinear interpolation or by cropping and padding
* save images slice by slice with or without masks of anatomical structure (set of structures).

First time the focus will be on datasets for cardiac image segmentation problem.

Currently the repository contains code for reading [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html). 
