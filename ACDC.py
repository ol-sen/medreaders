# Copyright (c) 2020 Olga Senyukova. All rights reserved.
# License: http://opensource.org/licenses/MIT
"""
medreaders --- Readers for medical image datasets
=================================================
**Source code:** `readers.py <https://github.com/ol-sen/medreaders/blob/master/readers.py>`_

--------------

The :mod:`readers` module contains the code for reading medical image datasets into memory and for auxiliary tasks:
    * resize images with ground truth masks by interpolation;
    * resize images with ground truth masks by cropping and padding;
    * save images slice by slice to PNG;
    * save pairs of images and ground truth masks to PNG.

You can specify anatomical structure of interest for loading ground truth masks or load masks with all structures.

You can use functions with default loader or create a custom loader.

In order to use the functions from this repository you should download a dataset that you need from `Grand Challenges in Biomedical Image Analysis <https://grand-challenge.org/challenges/>`_. Currently the repository contains the code for reading `ACDC dataset <https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html>`_.

Functions
---------

"""
import os
import glob
import numpy as np
import nibabel as nib
import imageio
import matplotlib.pyplot as plt
import sys
import logging
import skimage.transform


from functools import lru_cache


_acdc_readers = {}

class ACDC_Reader:
    def __init__(self):
        super().__init__()
        self.encoder = one_hot_encode
        self.decoder = one_hot_decode
        self.images = []
        self.masks = []

    def set_encoder(self, encode):
        self.encoder = encode

    def set_decoder(self, decode):
        self.decoder = decode

    def load(self, data_dir, structure, phase):
        """
        Loads ACDC dataset into memory (the dataset can be downloaded from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

        :param str data_dir: path to "training" directory
        :param str structure: anatomical structure of interest: right ventricle ("RV"), myocardium ("MYO"), left ventricle ("LV") or all structures ("all")
        :param str phase: end diastole ("ED"), end systole ("ES") or both phases ("both")
        :raises ValueError: if *mask* or *phase* parameters are incorrrect 
        :raises EnvironmentError: if the directory specified by *data_dir* parameter does not contain patient images 
         
        Example:

        .. code-block:: python
           
            custom_ACDC_Reader = readers.ACDC_Reader()
            custom_ACDC_Reader.set_encoder(readers.identity())
            custom_ACDC_Reader.set_decoder(readers.identity())
            custom_ACDC_Reader.load("../ACDC/training", "all", "ED")
        """
        if structure not in ["RV", "MYO", "LV", "all"]:
            raise ValueError("Incorrect 'mask' parameter in function 'load'. Expected values: 'RV', 'MYO', 'LV', 'all'.")
        if phase not in ["ED", "ES", "both"]:
            raise ValueError("Incorrect 'phase' parameter in function 'load'. Expected values: 'ED', 'ES', 'both'.")
        logging.info("Loading ACDC dataset...")
        glob_search = os.path.join(data_dir, "patient*")
        patient_dirs = sorted(glob.glob(glob_search))
        if len(patient_dirs) == 0:
            raise EnvironmentError("Loading ACDC failed. No patient directories were found in {}.".format(data_dir))
        del patient_dirs[1] # remove file patient001.Info.cfg from patient directories list
        self.images = [f for i in patient_dirs for f in load_patient_images(i, phase)]
        self.masks = [f for i in patient_dirs for f in self._load_patient_masks(i, structure, phase)]
        logging.info("ACDC dataset has been loaded successfully.")
         
    def resize(self, new_height, new_width, interpolate = True):
        """
        Resizes images according with their ground truth masks to (new_height, new_width). If *interpolate* parameter is set to **True**, bilinear interpolation for images and nearest neighbor interpolation for masks is used. Otherwise central cropping and/or zero padding is used for images and masks.

        :param new_height: height (in pixels) of output images and masks
        :type new_height: integer
        :param new_width: width (in pixels) of output images and masks
        :type new_width: integer
        :param interpolate: whether to use interpolation or cropping/padding
        :type interpolate: bool
        
        Example:

        .. code-block:: python
            
            custom_ACDC_Reader = readers.ACDC_Reader()
            custom_ACDC_Reader.set_encoder(readers.identity())
            custom_ACDC_Reader.set_decoder(readers.identity())
            custom_ACDC_Reader.load("../ACDC/training", "all", "ED")
            custom_ACDC_Reader.resize(216, 256) 
        
        Example:

        .. code-block:: python
            
            custom_ACDC_Reader.resize(216, 256, interpolate = False) 
        """    
        logging.info("Resizing images with masks...")
        if interpolate == True:
            self.images = [resize3D_image(new_height, new_width)(i)
                for i in self.images] 
            self.masks = [self._resize3D_mask(new_height, new_width)(m)
                for m in self.masks]
        else:
            self.images = [fit_to_box(new_height, new_width)(i)
                for i in self.images]
            self.masks = [self.encoder(fit_to_box(new_height, new_width)(self.decoder(m)))
                for m in self.masks]         
        logging.info("The images with masks have been resized successfully to {}x{}.".format(new_height, new_width))
    
    def save(self, save_dir_images, save_dir_masks, alpha = 0.5):
        """
        Saves original images slice by slice in PNG format to *save_dir_images*. Saves pairs of images and these images overlayed by ground truth masks slice by slice in PNG format to *save_dir_masks*.
        
        :param str save_dir_images: path to the directory for saving original images
        :param std save_dir_masks: path to the directory for saving images with masks
        :param alpha: the alpha blending value for mask overlay, between 0 (transparent) and 1 (opaque)
        :type alpha: float
        
        Example:

        .. code-block:: python
            
            custom_ACDC_Reader = readers.ACDC_Reader()
            custom_ACDC_Reader.set_encoder(readers.identity())
            custom_ACDC_Reader.set_decoder(readers.identity())
            custom_ACDC_Reader.load("../ACDC/training", "all", "ED")
            custom_ACDC_Reader.save("../PatientImagesOriginal", "../PatientImagesWithMasks")
        
        Example:

        .. code-block:: python
            
            custom_ACDC_Reader.save("../PatientImagesOriginal", "../PatientImagesWithMasks", alpha = 0.2)
"""   
        logging.info("Saving images...")
        os.mkdir(save_dir_images)
        for image_ind, image in enumerate(self.images):
            for slice_ind, slice_image in enumerate(normalize(slicing(image))):
                imageio.imwrite(
                    os.path.join(save_dir_images, "image{:03d}_slice{:02d}.png".format(image_ind + 1, slice_ind + 1)),
                    slice_image)
        logging.info("The images have been saved successfully to {} directory.".format(save_dir_images))
        
        def two_pictures_together_to_plot(image_slice, mask_slice):
            cmap_image = plt.cm.gray
            cmap_mask = plt.cm.Set1
            plt.figure(figsize = (8, 3.75))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.imshow(image_slice, cmap = cmap_image)
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.imshow(image_slice, cmap = cmap_image)
            plt.imshow(mask_slice, cmap = cmap_mask, alpha = alpha) 
        os.mkdir(save_dir_masks)
        for image_ind, (image, mask) in enumerate(zip(self.images, self.masks)):
            for slice_ind, (image_slice, mask_slice) in enumerate(zip(slicing(image), slicing(self.decoder(mask)))):
                two_pictures_together_to_plot(image_slice, mask_slice)
                path = os.path.join(save_dir_masks, "image{:03d}_slice{:02d}.png".format(image_ind + 1, slice_ind + 1)) 
                plt.savefig(path, bbox_inches = 'tight')
                plt.close()    
        logging.info("The images with masks have been saved successfully to {} directory.".format(save_dir_masks))

    def _load_patient_masks(self, patient_dir, structure, phase):
        return ([self.encoder(binarize_mask_if_one_structure(load_nifti_image(f), structure)) 
                for f in get_frames_paths(patient_dir, phase, create_frame_filename_mask)])

    @lru_cache()
    def _resize3D_mask(self, new_height, new_width):
        return lambda item: self.encoder(combine3D(
            map(truncate64,
            map(resize2D_mask(new_height, new_width),
                slicing(self.decoder(item))))))


def get_ACDC_reader(name):
    if name not in _acdc_readers:
        _acdc_readers[name] = ACDC_Reader()
    return _acdc_readers[name]


def set_encoder(encode):
    _default_ACDC_Reader.set_encoder(encode)


def set_decoder(decode):
    _default_ACDC_Reader.set_decoder(encode)


def load(data_dir, structure, phase):
    """
    Loads ACDC dataset into memory (the dataset can be downloaded from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

    :param str data_dir: path to "training" directory
    :param str structure: anatomical structure of interest: right ventricle ("RV"), myocardium ("MYO"), left ventricle ("LV") or all structures ("all")
    :param str phase: end diastole ("ED"), end systole ("ES") or both phases ("both")
    :raises ValueError: if *mask* or *phase* parameters are incorrrect 
    :raises EnvironmentError: if the directory specified by *data_dir* parameter does not contain patient images 
         
    Example:

    .. code-block:: python
           
        readers.load("../ACDC/training", "all", "ED")
    """
    return _default_ACDC_Reader.load(data_dir, structure, phase)


def resize(new_height, new_width, interpolate = True):
    """
    Resizes images according with their ground truth masks to (new_height, new_width). If *interpolate* parameter is set to **True**, bilinear interpolation for images and nearest neighbor interpolation for masks is used. Otherwise central cropping and/or zero padding is used for images and masks.

    :param new_height: height (in pixels) of output images and masks
    :type new_height: integer
    :param new_width: width (in pixels) of output images and masks
    :type new_width: integer
    :param interpolate: whether to use interpolation or cropping/padding
    :type interpolate: bool
        
    Example:

    .. code-block:: python
            
        readers.resize(216, 256) 
        
    Example:

    .. code-block:: python
            
        readers.resize(216, 256, interpolate = False) 
    """
    return _default_ACDC_Reader.resize(new_height, new_width, interpolate)


def save(save_dir_images, save_dir_masks, alpha = 0.5):
    """
    Saves original images slice by slice in PNG format to *save_dir_images*. Saves pairs of images and these images overlayed by ground truth masks slice by slice in PNG format to *save_dir_masks*.
        
    :param str save_dir_images: path to the directory for saving original images
    :param std save_dir_masks: path to the directory for saving images with masks
    :param alpha: the alpha blending value for mask overlay, between 0 (transparent) and 1 (opaque)
    :type alpha: float
        
    Example:

    .. code-block:: python
            
        readers.save("../PatientImagesOriginal", "../PatientImagesWithMasks")
        
    Example:

    .. code-block:: python
            
        readers.save("../PatientImagesOriginal", "../PatientImagesWithMasks", alpha = 0.2)
"""   
    return _default_ACDC_Reader.save(save_dir_images, save_dir_masks, alpha)


def get_images():
    """
    Get patient images.

    :return: images
    :rtype: list of numpy.ndarrays
    
    Example:

    .. code-block:: python

        images = readers.get_images()
    """
    return _default_ACDC_Reader.images


def get_masks():
    """
    Get ground truth masks of patient images.

    :return: ground truth masks
    :rtype: list of numpy.ndarrays
    
    Example:

    .. code-block:: python

        masks = readers.get_masks()
    """
    return _default_ACDC_Reader.masks


def one_hot_encode(mask):    
    """
    Performs one-hot encoding of ground truth mask.
    
    :param mask: input mask, each value of which is a class label, a number in range from 0 to *K*-1, where *K* is the total number of classes
    :type mask: numpy.ndarray
    :return: one-hot encoded mask
    :rtype: numpy.ndarray
    
    Example:

    .. code-block:: python
       
        encoded_mask = readers.one_hot_encode(input_mask)
    """
    input_shape = mask.shape
    mask = mask.ravel()
    classes = len(set(mask))
    n = len(mask)
    mask_encoded = np.zeros((n, classes))
    mask_encoded[np.arange(n), mask] = 1
    new_shape = input_shape + (classes,)
    mask_encoded = np.reshape(mask_encoded, new_shape)
    return mask_encoded


def one_hot_decode(mask):
    """
    Performs one-hot decoding of ground truth mask.
    
    :param mask: input mask, one-hot encoded
    :type mask: numpy.ndarray
    :return: output mask, each value of which is a class label, a number in range from 0 to *K*-1, where *K* is the total number of classes
    :rtype: numpy.ndarray
    
    Example:

    .. code-block:: python
       
        decoded_mask = readers.one_hot_decode(encoded_mask)
    """
    return mask.argmax(3)


def identity(mask):
    """
    This is identity function that can be used as an argument of functions :func:`set_encoder` and :func:`set_decoder`.
    
    :param mask: input mask
    :type mask: numpy.ndarray
    :return: the same mask
    :rtype: numpy.ndarray
    
    Example:

    .. code-block:: python
       
        custom_ACDC_Reader = readers.ACDC_Reader()
        custom_ACDC_Reader.set_encoder(readers.identity())
        custom_ACDC_Reader.set_decoder(readers.identity())
    """
    return mask


_default_ACDC_Reader = ACDC_Reader()


def height(item):
    return item.shape[0]


def width(item):
    return item.shape[1]


def depth(item):
    return item.shape[2]


def slicing(item):
    return (item[:, :, j] for j in range(depth(item)))


def combine3D(generator):
    slices = list(generator)
    dtype = type(slices[0][0][0])
    depth = len(slices)
    height, width = slices[0].shape
    new_item = np.zeros((height, width, depth), dtype = dtype)
    for i in range(depth):
        new_item[:, :, i] = slices[i]
    return new_item


def truncate16(item):
    return item.astype(np.int16)


def truncate64(item):
    return item.astype(np.int64)


@lru_cache()
def resize2D_image(new_height, new_width):
    return lambda item: skimage.transform.resize(
            item, 
            (new_height, new_width),
            clip = False,
            preserve_range = True)


@lru_cache()
def resize2D_mask(new_height, new_width):
    return lambda item: skimage.transform.resize(
            item, 
            (new_height, new_width),
            order = 0,
            preserve_range = True,
            anti_aliasing = False)


@lru_cache()
def resize3D_image(new_height, new_width):
    return lambda item: combine3D(
            map(truncate16,
            map(resize2D_image(new_height, new_width),
                slicing(item))))


def crop_height(item, new_height):
    remove_y_top = (height(item) - new_height) // 2
    remove_y_bottom = height(item) - new_height - remove_y_top;
    return skimage.util.crop(item, ((remove_y_top, remove_y_bottom), (0, 0), (0, 0)))


def crop_width(item, new_width):
    remove_x_left = (width(item) - new_width) // 2
    remove_x_right = width(item) - new_width - remove_x_left;
    return skimage.util.crop(item, ((0, 0), (remove_x_left, remove_x_right), (0, 0)))


def pad_height(item, new_height):
    add_y_top = (new_height - height(item)) // 2
    add_y_bottom = new_height - height(item) - add_y_top;
    return skimage.util.pad(item, ((add_y_top, add_y_bottom), (0, 0), (0, 0)), 'minimum')


def pad_width(item, new_width):
    add_x_left = (new_width - width(item)) // 2
    add_x_right = new_width - width(item) - add_x_left;
    return skimage.util.pad(item, ((0, 0), (add_x_left, add_x_right), (0, 0)), 'minimum')


@lru_cache()
def fit_to_box(new_height, new_width):
    def internal(item):
        if new_height < height(item):
            item = crop_height(item, new_height)
        if new_width < width(item):
            item = crop_width(item, new_width)
        if new_height > height(item):
            item = pad_height(item, new_height)
        if new_width > width(item):
            item = pad_width(item, new_width)
        return item
    return internal 


def normalize(slices):
    return ((((s - s.min()) / (s.max() - s.min()) * 255).astype('uint8')) 
            for s in slices)


def create_frame_filename_image(patient_dir, frame_ind):
    return "{}_frame{:02d}.nii.gz".format(os.path.split(patient_dir)[-1], frame_ind)


def create_frame_filename_mask(patient_dir, frame_ind):
    return "{}_frame{:02d}_gt.nii.gz".format(os.path.split(patient_dir)[-1], frame_ind)


def get_frames_paths(patient_dir, phase, create_frame_filename):
    info_file_path = os.path.join(patient_dir, "Info.cfg")
    if not os.path.isfile(info_file_path):
        raise EnvironmentError("Loadind ACDC failed. File Info.cfg was not found in patient directory {}.".format(patient_dir))

    with open(info_file_path, "r") as f:
        ED_info = f.readline().split(':')
        ED_frame_ind = int(ED_info[1].strip())
        ES_info = f.readline().split(':')
        ES_frame_ind = int(ES_info[1].strip())

    if phase == 'ED':
        frame_inds = [ED_frame_ind]
    elif phase == 'ES':
        frame_inds = [ES_frame_ind]
    elif phase == 'both':
        frame_inds = [ED_frame_ind, ES_frame_ind]

    return (os.path.join(patient_dir, create_frame_filename(patient_dir, i))
            for i in frame_inds)
   

def load_nifti_image(file_name):
    return truncate64(np.array(nib.load(file_name).get_fdata()))


def load_patient_images(patient_dir, phase):
    return ([load_nifti_image(f) for f in get_frames_paths(patient_dir, phase, create_frame_filename_image)])


def binarize_mask_if_one_structure(patient_mask, structure):
    """Noting to do if structure == 'all'"""
    bg_elems = []
    fg_elems = []
    if structure == 'RV':
        bg_elems = np.where(patient_mask != 1)
    elif structure == 'MYO':
        fg_elems = np.where(patient_mask == 2)
        bg_elems = np.where(patient_mask != 2)
    elif structure == 'LV':
        fg_elems = np.where(patient_mask == 3)
        bg_elems = np.where(patient_mask != 3)
    patient_mask[fg_elems] = 1
    patient_mask[bg_elems] = 0
    
    return patient_mask

