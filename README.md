# Readers for medical imaging datasets

The goal of this repository is to help researchers and practitioners working with medical imaging datasets to reduce an amount of routine work.

The repository contains the code for reading a dataset into memory and for auxiliary tasks:
* resize images with their ground truth masks;
* save images and their ground truth masks slice by slice.

In order to use the functions from this repository you should download a dataset that you need from [Grand Challenges in Biomedical Image Analysis](https://grand-challenge.org/challenges/).

First time the focus will be on datasets for cardiac image segmentation problem.

Currently the repository contains the code for reading [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

## Requirements

* Python 3.6.x

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
