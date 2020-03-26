# Readers for medical imaging datasets

The goal of this project is to help researchers and practitioners working with medical imaging datasets to reduce an amount of routine work.

The package contains the functions for reading a dataset into memory and for auxiliary tasks:
* resizing images with their ground truth masks;
* saving images and their ground truth masks slice by slice.

In order to use this package you should download a dataset that you need from [Grand Challenges in Biomedical Image Analysis](https://grand-challenge.org/challenges/).

First time the focus will be on datasets for cardiac image segmentation problem.

Currently the package provides the means for working with [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

## Requirements

* Python>=3.5

## Installation

```
pip3 install medreaders
```

## Documentation

Documentation is available at https://medreaders.readthedocs.io

## Project Structure
```
.
├── docs
│   ├── Makefile
│   ├── conf.py
│   ├── index.rst
│   └── make.bat
├── medreaders
│   ├── ACDC.py
│   └── __init__.py
├── tests
│   └── ACDC.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py

3 directories, 12 files
```

## Corresponding Author

* Olga Senyukova olga.senyukova@graphics.cs.msu.ru
