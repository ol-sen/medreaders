import os
import glob
import numpy as np
import nibabel as nib
import imageio
import matplotlib.pyplot as plt
import sys
import logging
import skimage

from keras import utils


def resize_images_with_masks(images, masks, new_height, new_width):
    """
    Resize images according with their masks to (new_height, new_width) using bilinear interpolation for images and nearest neighbor interpolation for masks
    Arguments:
        - images: list of 3D numpy arrays
        - masks: list of 4D numpy arrays corresponding to one-hot encoded masks for each image from images list
        - new_height: number of pixel rows in resized image
        - new_width: number of pixel columns in resized image
    Output:
        - two lists (images and masks) if the images with masks have been resized successfully:
        - two empty lists if the images with masks were not resized
    """   
    if not isinstance(images, list) or len(images) == 0 or not isinstance(images[0], np.ndarray) or len(images[0].shape) != 3:
        logging.error("Incorrect 'images' parameter in function 'save_images_with_masks'. 'Images' must be a nonempty list of 3D numpy arrays.")
        return [], []
    if not isinstance(masks, list) or len(masks) == 0 or not isinstance(masks[0], np.ndarray) or len(masks[0].shape) != 4:
        logging.error("Incorrect 'masks' parameter in function 'save_images_with_masks'. 'Masks' must be a nonempty list of 4D numpy arrays.")
        return [], []
 
    new_images = []
    new_masks = []
    for i in range(len(images)):
        height, width, depth = images[i].shape
        new_image = np.zeros((new_height, new_width, depth))
        for j in range(depth):
            cur_slice = images[i][:, :, j]
            new_slice = skimage.transform.resize(cur_slice, (new_height, new_width), clip = False, preserve_range = True)
            new_image[:, :, j] = new_slice.astype(np.int16)
        new_images += [new_image]
        
        height, width, depth, classes = masks[i].shape
        new_mask = np.zeros((new_height, new_width, depth))
        for j in range(depth):
            cur_slice_mask = masks[i][:, :, j, :]
            cur_slice_mask = cur_slice_mask.argmax(2)
            new_slice_mask = skimage.transform.resize(cur_slice_mask, (new_height, new_width), order = 0, preserve_range = True, anti_aliasing = False)
            new_mask[:, :, j] = new_slice_mask.astype(np.int64)
        new_classes = len(set(new_mask.flatten()))
        if new_classes < classes:
            logging.warning("Some structures have been lost after resizing image {}.".format(i + 1))
        new_mask = utils.to_categorical(new_mask).reshape(new_height, new_width, depth, new_classes)
        new_masks += [new_mask]

    logging.info("The images with masks have been resized successfully to {}x{}.".format(new_height, new_width))

    return new_images, new_masks


def crop_or_pad_images_with_masks(images, masks, new_height, new_width):
    """
    Central crop or zero pad images according with their masks in order to match (new_height, new_width)
    Arguments:
        - images: list of 3D numpy arrays
        - masks: list of 4D numpy arrays corresponding to one-hot encoded masks for each image from images list
        - new_height: number of pixel rows in resized image
        - new_width: number of pixel columns in resized image
    Output:
        - two lists (images and masks) if the images with masks have been resized successfully:
        - two empty lists if the images with masks were not resized
    """   
    if not isinstance(images, list) or len(images) == 0 or not isinstance(images[0], np.ndarray) or len(images[0].shape) != 3:
        logging.error("Incorrect 'images' parameter in function 'save_images_with_masks'. 'Images' must be a nonempty list of 3D numpy arrays.")
        return [], []
    if not isinstance(masks, list) or len(masks) == 0 or not isinstance(masks[0], np.ndarray) or len(masks[0].shape) != 4:
        logging.error("Incorrect 'masks' parameter in function 'save_images_with_masks'. 'Masks' must be a nonempty list of 4D numpy arrays.")
        return [], []
 
    new_images = []
    new_masks = []
    for i in range(len(images)):
        height, width, depth, classes = masks[i].shape
        new_image = images[i]
        new_mask = masks[i]
        if new_height < height:
            remove_y_top = (height - new_height) // 2
            remove_y_bottom = height - new_height - remove_y_top;
            new_image = skimage.util.crop(new_image, ((remove_y_top, remove_y_bottom), (0, 0), (0, 0)))
            new_mask = skimage.util.crop(new_mask, ((remove_y_top, remove_y_bottom), (0, 0), (0, 0), (0, 0)))
            height = new_height
        if new_width < width:
            remove_x_left = (width - new_width) // 2
            remove_x_right = width - new_width - remove_x_left;
            new_image = skimage.util.crop(new_image, ((0, 0), (remove_x_left, remove_x_right), (0, 0)))
            new_mask = skimage.util.crop(new_mask, ((0, 0), (remove_x_left, remove_x_right), (0, 0), (0, 0)))
            width = new_width
        if new_height > height:
            add_y_top = (new_height - height) // 2
            add_y_bottom = new_height - height - add_y_top;
            new_image = skimage.util.pad(new_image, ((add_y_top, add_y_bottom), (0, 0), (0, 0)), 'minimum')
            zero_mask = np.zeros((new_height, width, depth))
            new_mask_decoded = new_mask.argmax(3)
            zero_mask[add_y_bottom : add_y_bottom + height, :, :] = new_mask_decoded
            zero_mask = utils.to_categorical(zero_mask).reshape(new_height, width, depth, classes)
            new_mask = zero_mask
            height = new_height
        if new_width > width:
            add_x_left = (new_width - width) // 2
            add_x_right = new_width - width - add_x_left;
            new_image = skimage.util.pad(new_image, ((0, 0), (add_x_left, add_x_right), (0, 0)), 'minimum')
            zero_mask = np.zeros((height, new_width, depth))
            new_mask_decoded = new_mask.argmax(3)
            zero_mask[:, add_x_left : add_x_left + width, :] = new_mask_decoded
            zero_mask = utils.to_categorical(zero_mask).reshape(height, new_width, depth, classes)
            new_mask = zero_mask
        new_images += [new_image]
        new_masks += [new_mask]
         
    logging.info("The images with masks have been cropped or padded successfully to {}x{}.".format(new_height, new_width))

    return new_images, new_masks


def save_images(save_dir, images):
    """
    Save patient images slice by slice in PNG format
    Arguments:
        - save_dir: path to the directory for saving images
        - images: list of 3D numpy arrays
    Output:
        - 0 if the images were successfully saved to the specified directory, -1 otherwise
    """
    if not isinstance(images, list) or len(images) == 0 or not isinstance(images[0], np.ndarray) or len(images[0].shape) != 3:
        logging.error("Incorrect 'images' parameter in function 'save_images_with_masks'. 'Images' must be a nonempty list of 3D numpy arrays.")
        return -1
   
    try:
        os.mkdir(save_dir)
    except OSError:
        logging.error("Creation of the directory {} failed. The images have not been saved.".format(save_dir))
        return -1

    num_images = len(images)
    for image_ind in range(1, num_images + 1):
        img_dims = images[image_ind - 1].shape
        num_slices = img_dims[2]
        for slice_ind in range(1, num_slices + 1):
            cur_slice_image = images[image_ind - 1][:, :, slice_ind - 1]
            cur_slice_image = ((cur_slice_image - cur_slice_image.min()) / (cur_slice_image.max() - cur_slice_image.min()) * 255).astype('uint8')
            imageio.imwrite(os.path.join(save_dir, "image{:03d}_slice{:02d}.png".format(image_ind, slice_ind)), cur_slice_image)

    logging.info("The images have been saved successfully to {} directory.".format(save_dir))

    return 0


def save_images_with_masks(save_dir, images, masks, alpha = 0.5):
    """
    Save couples of patient image slices and these slices overlayed by masks with transparency specified by alpha (both original slices and slices with masks are displayed in one image)
    Arguments:
        - save_dir: path to the directory for saving images
        - images: list of 3D numpy arrays
        - masks: list of 4D numpy arrays corresponding to one-hot encoded masks for each image from images list
        - alpha: the alpha blending value for mask overlay, between 0 (transparent) and 1 (opaque)
    Output:
        - 0 if the images were successfully saved to the specified directory, -1 otherwise
    """
    if not isinstance(images, list) or len(images) == 0 or not isinstance(images[0], np.ndarray) or len(images[0].shape) != 3:
        logging.error("Incorrect 'images' parameter in function 'save_images_with_masks'. 'Images' must be a nonempty list of 3D numpy arrays.")
        return -1
    if not isinstance(masks, list) or len(masks) == 0 or not isinstance(masks[0], np.ndarray) or len(masks[0].shape) != 4:
        logging.error("Incorrect 'masks' parameter in function 'save_images_with_masks'. 'Masks' must be a nonempty list of 4D numpy arrays.")
        return -1
  
    try:
        os.mkdir(save_dir)
    except OSError:
        logging.error("Creation of the directory {} failed. The images with masks have not been saved.".format(save_dir))
        return -1
 
    num_images = len(images)
    for image_ind in range(1, num_images + 1):
        img_dims = images[image_ind - 1].shape
        num_slices = img_dims[2]
        for slice_ind in range(1, num_slices + 1):
            cur_slice_image = images[image_ind - 1][:, :, slice_ind - 1]
            cur_slice_mask = masks[image_ind - 1][:, :, slice_ind - 1, :]
            cur_slice_mask = cur_slice_mask.argmax(2)

            cmap_image = plt.cm.gray
            cmap_mask = plt.cm.Set1
            plt.figure(figsize = (8, 3.75))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.imshow(cur_slice_image, cmap = cmap_image)
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.imshow(cur_slice_image, cmap = cmap_image)
            plt.imshow(cur_slice_mask, cmap = cmap_mask, alpha = alpha)
            plt.savefig(os.path.join(save_dir, "image{:03d}_slice{:02d}.png".format(image_ind, slice_ind)), bbox_inches = 'tight')
            plt.close()
    
    logging.info("The images with masks have been saved successfully to {} directory.".format(save_dir))

    return 0


def load_ACDC(data_dir, mask = "RV", phase = "both"):
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
        - two empty lists if the dataset was not loaded
    """     
    if mask not in ["RV", "MYO", "LV", "all"]:
        logging.error("Incorrect 'mask' parameter in function 'load_ACDC'. Expected values: 'RV', 'MYO', 'LV', 'all'.")
        return [], []

    if phase not in ["ED", "ES", "both"]:
        logging.error("Incorrect 'phase' parameter in function 'load_ACDC'. Expected values: 'ED', 'ES', 'both'.")
        return [], []
        
    glob_search = os.path.join(data_dir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        logging.error("Loading ACDC failed. No patient directories were found in {}.".format(data_dir))
        return [], []

    del patient_dirs[1] # remove file patient001.Info.cfg from patient directories list
    
    images = []
    masks = []

    # load images and masks into memory
    patient_ind = 1;
    for patient_dir in patient_dirs:
        info_file_path = os.path.join(patient_dir, "Info.cfg")
        if not os.path.isfile(info_file_path):
            logging.error("Loadind ACDC failed. File Info.cfg was not found in patient directory {}.".format(patient_dir))
            return images, masks

        f = open(info_file_path, "r")

        lines = f.readlines()
        ED_info = lines[0].split(':')
        ED_frame_ind = int(ED_info[1].strip())
        ES_info = lines[1].split(':')
        ES_frame_ind = int(ES_info[1].strip())

        f.close()

        if phase == 'ED':
            frame_inds = [ED_frame_ind]
        elif phase == 'ES':
            frame_inds = [ES_frame_ind]
        elif phase == 'both':
            frame_inds = [ED_frame_ind, ES_frame_ind]

        for frame_ind in frame_inds:
            image_file_path = os.path.join(patient_dir, "patient{:03d}_frame{:02d}.nii.gz".format(patient_ind, frame_ind))
            patient_image_nifti = nib.load(image_file_path)
            patient_image = np.asarray(patient_image_nifti.get_data())
            images += [patient_image]

            mask_file_path = os.path.join(patient_dir, "patient{:03d}_frame{:02d}_gt.nii.gz".format(patient_ind, frame_ind))
            patient_mask_nifti = nib.load(mask_file_path)
            patient_mask = np.asarray(patient_mask_nifti.get_data())

            bg_elems = []
            fg_elems = []
            if mask == 'RV':
                bg_elems = np.where(patient_mask != 1)
            elif mask == 'MYO':
                fg_elems = np.where(patient_mask == 2)
                bg_elems = np.where(patient_mask != 2)
            elif mask == 'LV':
                fg_elems = np.where(patient_mask == 3)
                bg_elems = np.where(patient_mask != 3)

            patient_mask[fg_elems] = 1
            patient_mask[bg_elems] = 0

            masks += [patient_mask]

        patient_ind += 1

    for i in range(len(masks)):
        # one-hot encode masks
        dims = masks[i].shape
        num_classes = len(set(masks[i].flatten())) # get num classes from first image
        new_shape = dims + (num_classes,)
        masks[i] = utils.to_categorical(masks[i]).reshape(new_shape)

    logging.info("ACDC dataset has been loaded successfully.")
    
    return images, masks


