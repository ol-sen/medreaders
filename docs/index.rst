Medreaders --- Readers for medical imaging datasets
===================================================
**Source code:** https://github.com/ol-sen/medreaders

The package contains the functions for reading a dataset into memory and for auxiliary tasks:

    * resizing images with their ground truth masks;
    * saving images and their ground truth masks slice by slice.

In order to use this package you should download a dataset that you need from `Grand Challenges in Biomedical Image Analysis <https://grand-challenge.org/challenges/>`_.

Currently the package provides the means for working with `ACDC dataset <https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html>`_.

--------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
.. automodule:: ACDC
   :members: load,
             resize, 
             save,
             get_images,
             get_masks,
             set_encoder,
             set_decoder,
             one_hot_encode,
             one_hot_decode,
             identity


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
