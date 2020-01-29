# Copyright (c) 2019 Olga Senyukova. All rights reserved.
# License: http://opensource.org/licenses/MIT

import os
import glob
import numpy as np
import nibabel as nib
import imageio
import matplotlib.pyplot as plt
import sys
import logging
import skimage
import sklearn

from functools import lru_cache


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
    depth = len(slices)
    height, width = slices[0].shape
    new_item = np.zeros((height, width, depth))
    for i in range(depth):
        new_item[:, :, i] = slices[i]
    return new_item


def truncate16(item):
    return item.astype(np.int16)


def truncate64(item):
    return item.astype(np.int64)


def no_encode(mask):
    return mask


def no_decode(mask):
    return mask


def one_hot_encode(mask): 
    mask = mask.ravel()
    classes = len(set(mask))
    n = len(mask)
    mask_encoded = np.zeros((n, classes))
    mask_encoded[np.arange(n), mask] = 1
    new_shape = mask.shape + (classes,)
    mask_encoded = np.reshape(mask_encoded, new_shape)
    return mask_encoded


def one_hot_decode(mask):
    return mask.argmax(3)


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


@lru_cache()
def resize3D_mask(new_height, new_width, encode, decode):
    return lambda item: encode(combine3D(
            map(truncate64,
            map(resize2D_mask(new_height, new_width),
                slicing(decode(item))))))


def resize_images_with_masks(images, masks, new_height, new_width, encode, decode):
    """
    Resize images according with their masks to (new_height, new_width) using bilinear interpolation for images and nearest neighbor interpolation for masks
    Arguments:
        - images: list of 3D numpy arrays
        - masks: list of 4D numpy arrays corresponding to one-hot encoded masks for each image from images list
        - new_height: number of pixel rows in resized image
        - new_width: number of pixel columns in resized image
    Output:
        - resized images: list of 3D numpy arrays
        - resized masks: list of numpy arrays (3D - if encode = None and decode = None)
    """    
    logging.info("Resizing images with masks...")
    result = ([resize3D_image(new_height, new_width)(i) for i in images],
            [resize3D_mask(new_height, new_width, encode, decode)(m) for m in masks])
    logging.info("The images with masks have been resized successfully to {}x{}.".format(new_height, new_width))
    return result


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


def crop_or_pad_images_with_masks(images, masks, new_height, new_width, encode, decode):
    """
    Central crop or zero pad images according with their masks in order to match (new_height, new_width)
    Arguments:
        - images: list of 3D numpy arrays
        - masks: list of 4D numpy arrays corresponding to one-hot encoded masks for each image from images list
        - new_height: number of pixel rows in resized image
        - new_width: number of pixel columns in resized image
    Output:
        - two lists (images and masks) if the images with masks have been resized successfully:
    """    
    logging.info("Cropping or padding images with masks...")
    result = ([fit_to_box(new_height, new_width)(i) for i in images],
              [encode(fit_to_box(new_height, new_width)(decode(m))) for m in masks])         
    logging.info("The images with masks have been cropped or padded successfully to {}x{}.".format(new_height, new_width))
    return result


def normalize(slices):
    return ((((s - s.min()) / (s.max() - s.min()) * 255).astype('uint8')) 
            for s in slices)


def save_images(save_dir, images):
    """
    Save patient images slice by slice in PNG format
    Arguments:
        - save_dir: path to the directory for saving images
        - images: list of 3D numpy arrays
    """   
    logging.info("Saving images...")
    os.mkdir(save_dir)
    for image_ind, image in enumerate(images):
        for slice_ind, slice_image in enumerate(normalize(slicing(image))):
            imageio.imwrite(
                os.path.join(save_dir, "image{:03d}_slice{:02d}.png".format(image_ind + 1, slice_ind + 1)),
                slice_image)
    
    logging.info("The images have been saved successfully to {} directory.".format(save_dir))
 

def save_images_with_masks(save_dir, images, masks, alpha, decode):
    """
    Save couples of patient image slices and these slices overlayed by masks with transparency specified by alpha (both original slices and slices with masks are displayed in one image)
    Arguments:
        - save_dir: path to the directory for saving images
        - images: list of 3D numpy arrays
        - masks: list of 4D numpy arrays corresponding to one-hot encoded masks for each image from images list
        - alpha: the alpha blending value for mask overlay, between 0 (transparent) and 1 (opaque)
    """ 
    logging.info("Saving images with masks...")
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
     
    os.mkdir(save_dir)
    for image_ind, (image, mask) in enumerate(zip(images, masks)):
        for slice_ind, (image_slice, mask_slice) in enumerate(zip(slicing(image), slicing(decode(mask)))):
            two_pictures_together_to_plot(image_slice, mask_slice)
            path = os.path.join(save_dir, "image{:03d}_slice{:02d}.png".format(image_ind + 1, slice_ind + 1)) 
            plt.savefig(path, bbox_inches = 'tight')
            plt.close()    
    
    logging.info("The images with masks have been saved successfully to {} directory.".format(save_dir))


def create_frame_path_image(patient_dir, frame_ind):
    return "{}_frame{:02d}.nii.gz".format(os.path.split(patient_dir)[-1], frame_ind)


def create_frame_path_mask(patient_dir, frame_ind):
    return "{}_frame{:02d}_gt.nii.gz".format(os.path.split(patient_dir)[-1], frame_ind)


def get_frames_paths(patient_dir, phase, create_frame_path):
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

    return (os.path.join(patient_dir, create_frame_path(patient_dir, i))
            for i in frame_inds)
   

def load_nifti_image(file_name):
    return np.asarray(nib.load(file_name).get_data())


def load_patient_images(patient_dir, phase):
    return ([load_nifti_image(f) for f in get_frames_paths(patient_dir, phase, create_frame_path_image)])


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


def load_patient_masks(patient_dir, mask, phase, encode):
    return ([encode(binarize_mask_if_one_structure(load_nifti_image(f), mask)) 
            for f in get_frames_paths(patient_dir, phase, create_frame_path_mask)])
    

def load_ACDC(data_dir, mask, phase, encode):
    """
    Load ACDC dataset into memory (dataset can be downloaded from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html)
    Arguments:
        - data_dir: path to "training" directory
        - mask: anatomical structure of interest - right ventricle ("RV"), myocardium ("MYO"), left ventricle ("LV"), all structures ("all")
        - phase: end diastole ("ED"), end systole ("ES"),  both phases ("both")
    Output:
        - two lists (images and masks) if the dataset was loaded successfully:
            - images: list of 3D numpy arrays, loaded from .nii.gz files in "patientXXX" directories, corresponding to end diastole phase (if phase = "ED"), end systole phase (if phase = "ES") or both phases (if phase = "both")
            - masks: list of 4D numpy arrays corresponding to one-hot encoded masks for each image from images output list (if mask = "RV", "MYO" or "LV", array values are [0., 1.] ("1" is encoded) for structure voxels and [1., 0.] ("0" is encoded) for background voxels, if mask = "all", array values are [1., 0., 0., 0.] ("0") for background, [0., 1., 0., 0.] ("1") for right vectricle, [0., 0., 1., 0.] ("2") for myocardium, [0., 0., 0., 1.] ("3") for left ventricle)
    """     
    if mask not in ["RV", "MYO", "LV", "all"]:
        raise ValueError("Incorrect 'mask' parameter in function 'load_ACDC'. Expected values: 'RV', 'MYO', 'LV', 'all'.")

    if phase not in ["ED", "ES", "both"]:
        raise ValueError("Incorrect 'phase' parameter in function 'load_ACDC'. Expected values: 'ED', 'ES', 'both'.")
        
    logging.info("Loading ACDC dataset...")
    glob_search = os.path.join(data_dir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise EnvironmentError("Loading ACDC failed. No patient directories were found in {}.".format(data_dir))

    del patient_dirs[1] # remove file patient001.Info.cfg from patient directories list
    
    result = ([f for i in patient_dirs for f in load_patient_images(i, phase)],
              [f for i in patient_dirs for f in load_patient_masks(i, mask, phase, encode)])         
    logging.info("ACDC dataset has been loaded successfully.")
    return result
