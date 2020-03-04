import numpy as np
import logging
import unittest

import sys
sys.path.append("/project")
from medreaders import ACDC

logging.basicConfig(level = logging.INFO)


class TestLoad(unittest.TestCase):
    def test_RV_ED(self):
        ACDC.load("datasets_samples/ACDC", "RV", "ED")
        assert len(ACDC.get_images()) == 2
        assert len(ACDC.get_masks()) == 2
 
    def test_RV_both(self):
        ACDC.load("datasets_samples/ACDC", "RV", "both")
        assert len(ACDC.get_images()) == 4
        assert len(ACDC.get_masks()) == 4


class TestResize(unittest.TestCase):
    def setUp(self):
        ACDC.load("datasets_samples/ACDC", "all", "ED")
    
    def test_300x300_Interpolate(self):
        ACDC.resize(300, 300)
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert all(i.shape[0] == 300 and i.shape[1] == 300 for i in images)
        assert all(m.shape[0] == 300 and m.shape[1] == 300 for m in masks)

    def test_300x300_Interpolate(self):
        ACDC.resize(300, 300, interpolate = False)
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert all(i.shape[0] == 300 and i.shape[1] == 300 for i in images)
        assert all(m.shape[0] == 300 and m.shape[1] == 300 for m in masks)


class TestSave(unittest.TestCase):
    def setUp(self):
        ACDC.load("datasets_samples/ACDC", "all", "ED")
     
    def test_default(self):
        ACDC.save("Images1", "Masks1")    

    def test_default_alpha_1(self):
        ACDC.save("Images2", "Masks2", alpha = 1) 


class TestOneHotEncode(unittest.TestCase):
    def test_012x210(self):
        mask = np.array([[0, 1, 2], [2, 1, 0]])
        mask_encoded = ACDC.one_hot_encode(mask)
        assert (mask_encoded == np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                          [[0, 0, 1], [0, 1, 0], [1, 0, 0]]])).all()
        assert type(mask[0][0]) == type(mask_encoded[0][0][0])


class TestOneHotDecode(unittest.TestCase):
    def test_012x210(self):
        mask_encoded = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 [[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
        mask_decoded = ACDC.one_hot_decode(mask_encoded)
        assert (mask_decoded == np.array([[0, 1, 2], [2, 1, 0]])).all()



class TestLoadPatientMasks(unittest.TestCase):
    def test_RV_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "RV", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_MYO_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "MYO", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_LV_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "LV", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_all_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "all", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_RV_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "RV", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_MYO_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "MYO", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_LV_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "LV", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_all_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "all", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_RV_both(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "RV", "both")
        masks = list(masks_generator)
        assert len(masks) == 2

    def test_MYO_both(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "MYO", "both")
        masks = list(masks_generator)
        assert len(masks) == 2

    def test_LV_both(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "LV", "both")
        masks = list(masks_generator)
        assert len(masks) == 2

    def test_all_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("datasets_samples/ACDC/patient001", "all", "both")
        masks = list(masks_generator)
        assert len(masks) == 2

class TestResize3DMask(unittest.TestCase):
    def test_3x3x3_2x2_one_hot(self):
        mask = np.zeros((3, 3, 3), dtype = int)
        mask[0][0][0] = 1
        mask_encoded = ACDC.one_hot_encode(mask)
        result = ACDC._default_ACDC_Reader._resize3D_mask(2, 2)(mask_encoded)
        assert result.shape == (2, 2, 3, 2)

def test_3x3x3_2x2_identity(self):
        mask = np.zeros((3, 3, 3), dtype = int)
        mask[0][0][0] = 1
        ACDC.custom_ACDC_Reader()
        ACDC.custom_ACDC_Reader.set_encoder(ACDC.identity)
        ACDC.custom_ACDC_Reader.set_decoder(ACDC.identity)
        result = ACDC.custom_ACDC_Reader._resize3D_mask(2, 2)(mask_encoded)
        assert result.shape == (2, 2, 3)


class TestCombine3D(unittest.TestCase):
    def test_2x2x2(self):
        slice1 = np.array([[1, 2], [3, 4]])
        slice2 = np.array([[5, 6], [7, 8]])
        generator = (slice1, slice2)

        result = ACDC.combine3D(generator)
        
        ground_truth = np.zeros((2, 2, 2), dtype = int)
        start = 1
        for k in range(2): #depth
            for i in range(2): #height
                for j in range(2): #width
                    ground_truth[i][j][k] = start
                    start += 1
        
        ground_truth = np.array(ground_truth)
        assert result.shape == (2, 2, 2)
        assert (result == ground_truth).all()
        assert type(result[0][0][0]) == type(slice1[0][0])


class TestResize3DImage(unittest.TestCase):
    def test_3x3x3_2x2(self):
        image = np.zeros((3, 3, 3), dtype = int)
        result = ACDC.resize3D_image(2, 2)(image)
        assert result.shape == (2, 2, 3)


class TestCropHeight(unittest.TestCase):
    def test_3x3x3_2(self):
        image = np.zeros((3, 3, 3), dtype = int)
        result = ACDC.crop_height(image, 2)
        assert result.shape == (2, 3, 3)
    
    def test_4x4x4_2(self):
        image = np.zeros((4, 4, 4), dtype = int)
        result = ACDC.crop_height(image, 2)
        assert result.shape == (2, 4, 4)


class TestCropWidth(unittest.TestCase):
    def test_3x3x3_2(self):
        image = np.zeros((3, 3, 3), dtype = int)
        result = ACDC.crop_width(image, 2)
        assert result.shape == (3, 2, 3)
    
    def test_4x4x4_2(self):
        image = np.zeros((4, 4, 4), dtype = int)
        result = ACDC.crop_width(image, 2)
        assert result.shape == (4, 2, 4)


class TestPadHeight(unittest.TestCase):
    def test_2x2x2_3(self):
        image = np.zeros((2, 2, 2), dtype = int)
        result = ACDC.pad_height(image, 3)
        assert result.shape == (3, 2, 2)
    
    def test_2x2x2_4(self):
        image = np.zeros((2, 2, 2), dtype = int)
        result = ACDC.pad_height(image, 4)
        assert result.shape == (4, 2, 2)


class TestPadWidth(unittest.TestCase):
    def test_2x2x2_3(self):
        image = np.zeros((2, 2, 2), dtype = int)
        result = ACDC.pad_width(image, 3)
        assert result.shape == (2, 3, 2)
    
    def test_2x2x2_4(self):
        image = np.zeros((2, 2, 2), dtype = int)
        result = ACDC.pad_width(image, 4)
        assert result.shape == (2, 4, 2)


class TestFitToBox(unittest.TestCase):
    def test_3x3x3_2x2(self):
        image = np.zeros((3, 3, 3), dtype = int)
        result = ACDC.fit_to_box(2, 2)(image)
        assert result.shape == (2, 2, 3)

    def test_3x3x3_4x2(self):
        image = np.zeros((3, 3, 3), dtype = int)
        result = ACDC.fit_to_box(4, 2)(image)
        assert result.shape == (4, 2, 3)

    def test_3x3x3_2x4(self):
        image = np.zeros((3, 3, 3), dtype = int)
        result = ACDC.fit_to_box(2, 4)(image)
        assert result.shape == (2, 4, 3)

    def test_3x3x3_4x4(self):
        image = np.zeros((3, 3, 3), dtype = int)
        result = ACDC.fit_to_box(4, 4)(image)
        assert result.shape == (4, 4, 3)


class TestNormalize(unittest.TestCase):
    def test_0112_2110(self):
        slice1 = np.array([0, 1, 1])
        slice2 = np.array([0, 1, 2])
        generator = (slice1, slice2)
        result_generator = ACDC.normalize(generator)
        result = list(result_generator)
        assert (result[0] == np.array([0, 255, 255])).all()
        assert (result[1] == np.array([0, 127, 255])).all()


class TestCreateFrameFilenameImage(unittest.TestCase):
    def test_patient001_frame01(self):
        filename = ACDC.create_frame_filename_image("datasets_samples/ACDC/patient001", 1)
        assert filename == "patient001_frame01.nii.gz"        


class TestCreateFrameFilenameMask(unittest.TestCase):
    def test_patient001_frame12(self):
        filename = ACDC.create_frame_filename_mask("datasets_samples/ACDC/patient001", 12)
        assert filename == "patient001_frame12_gt.nii.gz"        


class TestGetFramesPaths(unittest.TestCase):
    def test_patient001_ED_image(self):
        result_generator = ACDC.get_frames_paths("datasets_samples/ACDC/patient001", "ED", ACDC.create_frame_filename_image)
        result = list(result_generator)
        assert result[0] == "datasets_samples/ACDC/patient001/patient001_frame01.nii.gz"    

    def test_patient001_ED_mask(self):
        result_generator = ACDC.get_frames_paths("datasets_samples/ACDC/patient001", "ED", ACDC.create_frame_filename_mask)
        result = list(result_generator)
        assert result[0] == "datasets_samples/ACDC/patient001/patient001_frame01_gt.nii.gz" 

    def test_patient001_ES_image(self):
        result_generator = ACDC.get_frames_paths("datasets_samples/ACDC/patient001", "ES", ACDC.create_frame_filename_image)
        result = list(result_generator)
        assert result[0] == "datasets_samples/ACDC/patient001/patient001_frame12.nii.gz"    

    def test_patient001_both_image(self):
        result_generator = ACDC.get_frames_paths("datasets_samples/ACDC/patient001", "both", ACDC.create_frame_filename_image)
        result = list(result_generator)
        assert result[0] == "datasets_samples/ACDC/patient001/patient001_frame01.nii.gz"    
        assert result[1] == "datasets_samples/ACDC/patient001/patient001_frame12.nii.gz"    


class TestLoadNiftiImage(unittest.TestCase):
    def test_patient001(self):
        image = ACDC.load_nifti_image("datasets_samples/ACDC/patient001/patient001_frame01.nii.gz")
        assert type(image) == np.ndarray


class TestLoadPatientImages(unittest.TestCase):
    def test_patient001_ED(self):
        images_generator = ACDC.load_patient_images("datasets_samples/ACDC/patient001", "ED")
        images = list(images_generator)
        assert len(images) == 1 

    def test_patient001_ES(self):
        images_generator = ACDC.load_patient_images("datasets_samples/ACDC/patient001", "ES")
        images = list(images_generator)
        assert len(images) == 1 

    def test_patient001_both(self):
        images_generator = ACDC.load_patient_images("datasets_samples/ACDC/patient001", "both")
        images = list(images_generator)
        assert len(images) == 2


class TestBinarizeMaskIfOneStructure(unittest.TestCase):
    def setUp(self):
        self.mask = np.array([0, 1, 2, 3])
    
    def test_all(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "all")
        assert (result == self.mask).all()

    def test_RV(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "RV")
        assert (result == np.array([0, 1, 0, 0])).all()   

    def test_MYO(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "MYO")
        assert (result == np.array([0, 0, 1, 0])).all()   

    def test_RV(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "LV")
        assert (result == np.array([0, 0, 0, 1])).all()

