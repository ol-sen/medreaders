import numpy as np
import logging
import unittest
import os

from medreaders import ACDC

logging.basicConfig(level = logging.INFO)


class TestAll(unittest.TestCase):
    def test_default(self):
        if not os.path.isdir("results"):
            os.mkdir("results")
        ACDC.load("../../datasets_samples/ACDC", "all", "ED")        
        ACDC.resize(216, 256) 
        ACDC.save("results/PatientImages", "results/PatientImagesWithMasks")
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert len(images) == 2
        assert len(masks) == 2
        assert images[0].shape == (10, 1, 216, 256)
      

class TestSetEncoderDecoder(unittest.TestCase):
    def setUp(self):
        if not os.path.isdir("results"):
            os.mkdir("results")

    def test_one_hot_encode(self):
        ACDC.set_encoder(ACDC.one_hot_encode)
        ACDC.set_decoder(ACDC.one_hot_decode)
        ACDC.load("../../datasets_samples/ACDC", "all", "ED")        
        ACDC.resize(216, 256) 
        ACDC.save("results/PatientImagesOneHot", "results/PatientImagesWithMasksOneHot")
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert len(images) == 2
        assert len(masks) == 2
        assert images[0].shape == (10, 1, 216, 256)
    
    def test_identity(self):
        ACDC.set_encoder(ACDC.identity)
        ACDC.set_decoder(ACDC.identity)
        ACDC.load("../../datasets_samples/ACDC", "all", "ED")        
        ACDC.resize(216, 256) 
        ACDC.save("results/PatientImagesIdentity", "results/PatientImagesWithMasksIdentity")
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert len(images) == 2
        assert len(masks) == 2
        assert images[0].shape == (10, 1, 216, 256)

    def tearDown(self):
        ACDC.set_encoder(ACDC.one_hot_encode)
        ACDC.set_decoder(ACDC.one_hot_decode)


class TestSetImagesFormat(unittest.TestCase):
    def setUp(self):
        if not os.path.isdir("results"):
            os.mkdir("results")
    
    def test_PyTorch(self):
        ACDC.set_images_format("PyTorch")
        ACDC.load("../../datasets_samples/ACDC", "all", "ED")        
        ACDC.resize(216, 256) 
        ACDC.save("results/PatientImagesPyTorch", "results/PatientImagesWithMasksPyTorch")
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert len(images) == 2
        assert len(masks) == 2
        assert images[0].shape == (10, 1, 216, 256)

    def test_Keras(self):
        ACDC.set_images_format("Keras")
        ACDC.load("../../datasets_samples/ACDC", "all", "ED")        
        ACDC.resize(216, 256) 
        ACDC.save("results/PatientImagesKeras", "results/PatientImagesWithMasksKeras")
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert len(images) == 2
        assert len(masks) == 2
        assert images[0].shape == (10, 216, 256, 1)

    def test_incorrect_images_format(self):
        with self.assertRaises(ValueError):
            ACDC.set_images_format("aaa")

    def tearDown(self):
        ACDC.set_images_format("PyTorch")


class TestACDCReaderLoad(unittest.TestCase):
    def test_RV_ED(self):
        ACDC._default_ACDC_Reader.load("../../datasets_samples/ACDC", "RV", "ED")
        assert len(ACDC._default_ACDC_Reader.get_images()) == 2
        assert len(ACDC._default_ACDC_Reader.get_masks()) == 2
 
    def test_RV_both(self):
        ACDC._default_ACDC_Reader.load("../../datasets_samples/ACDC", "RV", "both")
        assert len(ACDC._default_ACDC_Reader.get_images()) == 4
        assert len(ACDC._default_ACDC_Reader.get_masks()) == 4

    def test_incorrect_structure(self):
        with self.assertRaises(ValueError):
            ACDC._default_ACDC_Reader.load("../../datasets_samples/ACDC", "aaa", "ED")
    
    def test_incorrect_phase(self):
        with self.assertRaises(ValueError):
            ACDC._default_ACDC_Reader.load("../../datasets_samples/ACDC", "RV", "aaa")
    
    def test_incorrect_directory(self):
        with self.assertRaises(EnvironmentError):
            ACDC._default_ACDC_Reader.load("../../datasets_samples", "RV", "ED")
 

class TestACDCReaderResize(unittest.TestCase):
    def setUp(self):
        ACDC._default_ACDC_Reader.load("../../datasets_samples/ACDC", "all", "ED")
        if not os.path.isdir("results"):
            os.mkdir("results")
    
    def test_300x300_interpolate(self):
        ACDC._default_ACDC_Reader.resize(300, 300)
        ACDC._default_ACDC_Reader.save("results/Images300x300Interp", "results/Masks300x300Interp")
        images = ACDC._default_ACDC_Reader.get_images()
        masks = ACDC._default_ACDC_Reader.get_masks()
        fmt = ACDC._default_ACDC_Reader.get_images_format()
        assert all(i.shape[2] == 300 and i.shape[3] == 300 for i in images)
        assert all(m.shape[1] == 300 and m.shape[2] == 300 for m in masks)

    def test_300x300_no_interpolate(self):
        ACDC._default_ACDC_Reader.resize(300, 300, interpolate = False)
        ACDC._default_ACDC_Reader.save("results/Images300x300NoInterp", "results/Masks300x300NoInterp")
        images = ACDC._default_ACDC_Reader.get_images()
        masks = ACDC._default_ACDC_Reader.get_masks()
        fmt = ACDC._default_ACDC_Reader.get_images_format()
        assert all(i.shape[2] == 300 and i.shape[3] == 300 for i in images)
        assert all(m.shape[1] == 300 and m.shape[2] == 300 for m in masks)

    def test_100x100_interpolate(self):
        ACDC._default_ACDC_Reader.resize(100, 100)
        ACDC._default_ACDC_Reader.save("results/Images100x100Interp", "results/Masks100x100Interp")
        images = ACDC._default_ACDC_Reader.get_images()
        masks = ACDC._default_ACDC_Reader.get_masks()
        fmt = ACDC._default_ACDC_Reader.get_images_format()
        assert all(i.shape[2] == 100 and i.shape[3] == 100 for i in images)
        assert all(m.shape[1] == 100 and m.shape[2] == 100 for m in masks)

    def test_100x100_no_interpolate(self):
        ACDC._default_ACDC_Reader.resize(100, 100, interpolate = False)
        ACDC._default_ACDC_Reader.save("results/Images100x100NoInterp", "results/Masks100x100NoInterp")
        images = ACDC._default_ACDC_Reader.get_images()
        masks = ACDC._default_ACDC_Reader.get_masks()
        fmt = ACDC._default_ACDC_Reader.get_images_format()
        assert all(i.shape[2] == 100 and i.shape[3] == 100 for i in images)
        assert all(m.shape[1] == 100 and m.shape[2] == 100 for m in masks)


class TestACDCReaderSave(unittest.TestCase):
    def setUp(self):
        ACDC._default_ACDC_Reader.load("../../datasets_samples/ACDC", "all", "ED")
        if not os.path.isdir("results"):
            os.mkdir("results")

    def test_default(self):
        ACDC._default_ACDC_Reader.save("results/ImagesAlphaDefault", "results/MasksAlphaDefault")    

    def test_default_alpha_1(self):
        ACDC._default_ACDC_Reader.save("results/ImagesAlpha1", "results/MasksAlpha1", alpha = 1) 


class TestACDCReaderReshapeImageToFormat(unittest.TestCase):
    def test_2x3x4_PyTorch(self):
        image = np.zeros((2, 3, 4))
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        image = ACDC._default_ACDC_Reader._reshape_image_to_format(image)
        assert image.shape == (2, 1, 3, 4)
   
    def test_2x3x4_Keras(self):
        image = np.zeros((2, 3, 4))
        ACDC._default_ACDC_Reader.set_images_format("Keras")
        image = ACDC._default_ACDC_Reader._reshape_image_to_format(image)
        assert image.shape == (2, 3, 4, 1)
    
    def tearDown(self):    
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")


class TestACDCReaderLoadPatientImages(unittest.TestCase):
    def test_patient001_ED(self):
        images_generator = ACDC._default_ACDC_Reader._load_patient_images("../../datasets_samples/ACDC/patient001", "ED")
        images = list(images_generator)
        assert len(images) == 1 

    def test_patient001_ES(self):
        images_generator = ACDC._default_ACDC_Reader._load_patient_images("../../datasets_samples/ACDC/patient001", "ES")
        images = list(images_generator)
        assert len(images) == 1 

    def test_patient001_both(self):
        images_generator = ACDC._default_ACDC_Reader._load_patient_images("../../datasets_samples/ACDC/patient001", "both")
        images = list(images_generator)
        assert len(images) == 2


class TestACDCReaderLoadPatientMasks(unittest.TestCase):
    def test_RV_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "RV", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_MYO_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "MYO", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_LV_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "LV", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_all_ED(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "all", "ED")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_RV_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "RV", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_MYO_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "MYO", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_LV_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "LV", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_all_ES(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "all", "ES")
        masks = list(masks_generator)
        assert len(masks) == 1

    def test_RV_both(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "RV", "both")
        masks = list(masks_generator)
        assert len(masks) == 2

    def test_MYO_both(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "MYO", "both")
        masks = list(masks_generator)
        assert len(masks) == 2

    def test_LV_both(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "LV", "both")
        masks = list(masks_generator)
        assert len(masks) == 2

    def test_all_both(self):
        masks_generator = ACDC._default_ACDC_Reader._load_patient_masks("../../datasets_samples/ACDC/patient001", "all", "both")
        masks = list(masks_generator)
        assert len(masks) == 2


class TestACDCReaderTransposeImageToSkimage(unittest.TestCase):
    def test_2x1x3x4_PyTorch(self):
        image = np.zeros((2, 1, 3, 4))
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        image = ACDC._default_ACDC_Reader._transpose_image_to_skimage(image)
        assert image.shape == (2, 3, 4, 1)
    
    def test_2x3x4x1_Keras(self):
        image = np.zeros((2, 3, 4, 1))
        ACDC._default_ACDC_Reader.set_images_format("Keras")
        image = ACDC._default_ACDC_Reader._transpose_image_to_skimage(image)
        assert image.shape == (2, 3, 4, 1)
    
    def tearDown(self):    
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")


class TestACDCReaderTransposeImageFromSkimage(unittest.TestCase):
    def test_2x3x4x1_PyTorch(self):
        image = np.zeros((2, 3, 4, 1))
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        image = ACDC._default_ACDC_Reader._transpose_image_from_skimage(image)
        assert image.shape == (2, 1, 3, 4)
    
    def test_2x3x4x1_Keras(self):
        image = np.zeros((2, 3, 4, 1))
        ACDC._default_ACDC_Reader.set_images_format("Keras")
        image = ACDC._default_ACDC_Reader._transpose_image_from_skimage(image)
        assert image.shape == (2, 3, 4, 1)

    def tearDown(self):    
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")


class TestACDCReaderResize3DImage(unittest.TestCase):
    def test_PyTorch_3x1x3x3_2x2(self):
        image = np.zeros((3, 1, 3, 3), dtype = int) 
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        result = ACDC._default_ACDC_Reader._resize3D_image(2, 2)(image)
        assert result.shape == (3, 1, 2, 2)
    
    def test_Keras_3x3x3x1_2x2(self):
        image = np.zeros((3, 3, 3, 1), dtype = int)
        ACDC._default_ACDC_Reader.set_images_format("Keras")
        result = ACDC._default_ACDC_Reader._resize3D_image(2, 2)(image)
        assert result.shape == (3, 2, 2, 1)

    def tearDown(self):    
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")

    
class TestACDCReaderResize3DMask(unittest.TestCase):
    def test_3x3x3_2x2_one_hot(self):
        mask = np.zeros((3, 3, 3), dtype = int)
        mask[0][0][0] = 1
        ACDC._default_ACDC_Reader.set_encoder(ACDC.one_hot_encode)
        ACDC._default_ACDC_Reader.set_decoder(ACDC.one_hot_decode)
        mask_encoded = ACDC.one_hot_encode(mask)
        result = ACDC._default_ACDC_Reader._resize3D_mask(2, 2)(mask_encoded)
        assert result.shape == (3, 2, 2, 2)

    def test_3x3x3_2x2_identity(self):
        mask = np.zeros((3, 3, 3), dtype = int)
        mask[0][0][0] = 1
        ACDC._default_ACDC_Reader.set_encoder(ACDC.identity)
        ACDC._default_ACDC_Reader.set_decoder(ACDC.identity)
        result = ACDC._default_ACDC_Reader._resize3D_mask(2, 2)(mask)
        assert result.shape == (3, 2, 2)

    def tearDown(self):
        ACDC._default_ACDC_Reader.set_encoder(ACDC.one_hot_encode)
        ACDC._default_ACDC_Reader.set_decoder(ACDC.one_hot_decode)


class TestACDCReaderFitToBox(unittest.TestCase):
    def test_Keras_3x3x3x1_2x2(self):
        image = np.zeros((3, 3, 3, 1), dtype = int)
        ACDC._default_ACDC_Reader.set_images_format("Keras")
        result = ACDC._default_ACDC_Reader._fit_to_box(2, 2)(image)
        assert result.shape == (3, 2, 2, 1)

    def test_PyTorch_3x1x3x3_2x2(self):
        image = np.zeros((3, 1, 3, 3), dtype = int)
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        result = ACDC._default_ACDC_Reader._fit_to_box(2, 2)(image)
        assert result.shape == (3, 1, 2, 2)

    def test_PyTorch_3x1x3x3_4x2(self):
        image = np.zeros((3, 1, 3, 3), dtype = int) 
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        result = ACDC._default_ACDC_Reader._fit_to_box(4, 2)(image)
        assert result.shape == (3, 1, 4, 2)

    def test_PyTorch_3x1x3x3_2x4(self):
        image = np.zeros((3, 1, 3, 3), dtype = int) 
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        result = ACDC._default_ACDC_Reader._fit_to_box(2, 4)(image)
        assert result.shape == (3, 1, 2, 4)

    def test_PyTorch_3x1x3x3_4x4(self):
        image = np.zeros((3, 1, 3, 3), dtype = int)
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")
        result = ACDC._default_ACDC_Reader._fit_to_box(4, 4)(image)
        assert result.shape == (3, 1, 4, 4)

    def test_mask_3x3x3_2x2(self):
        mask = np.zeros((3, 3, 3), dtype = int)
        result = ACDC._default_ACDC_Reader._fit_to_box(2, 2)(mask)
        assert result.shape == (3, 2, 2)
    
    def tearDown(self):
        ACDC._default_ACDC_Reader.set_images_format("PyTorch")


class TestLoad(unittest.TestCase):
    def test_RV_ED(self):
        ACDC.load("../../datasets_samples/ACDC", "RV", "ED")
        assert len(ACDC.get_images()) == 2
        assert len(ACDC.get_masks()) == 2


class TestResize(unittest.TestCase):
    def test_300x300(self): 
        ACDC.load("../../datasets_samples/ACDC", "all", "ED")
        ACDC.resize(300, 300)
        images = ACDC.get_images()
        masks = ACDC.get_masks()
        assert all(i.shape[2] == 300 and i.shape[3] == 300 for i in images)
        assert all(m.shape[1] == 300 and m.shape[2] == 300 for m in masks)
 

class TestSave(unittest.TestCase):
    def test_default(self):
        ACDC.load("../../datasets_samples/ACDC", "all", "ED")
        if not os.path.isdir("results"):
            os.mkdir("results")
        ACDC.save("results/ImagesAlphaDefault_2", "results/MasksAlphaDefault_2")    

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


class TestCombine3DImage(unittest.TestCase):
    def test_2x2x2x1(self):
        slice1 = np.array([[[0], [1], [0], [1]], [[0], [2], [0], [2]]])
        slice2 = np.array([[[1], [0], [1], [0]], [[2], [0], [2], [0]]])
        slice3 = np.array([[[1], [1], [1], [1]], [[2], [2], [2], [2]]])

        generator = (slice1, slice2, slice3)

        result = ACDC.combine3D_image(generator)
        
        assert result.shape == (3, 2, 4, 1)
        assert type(result[0][0][0][0]) == type(slice1[0][0][0])


class TestCombine3DMask(unittest.TestCase):
    def test_2x2x2(self):
        slice1 = np.array([[0, 1, 0, 1], [0, 2, 0, 2]])
        slice2 = np.array([[1, 0, 1, 0], [2, 0, 2, 0]])
        slice3 = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])

        generator = (slice1, slice2, slice3)

        result = ACDC.combine3D_mask(generator)
        
        assert result.shape == (3, 2, 4)
        assert type(result[0][0][0]) == type(slice1[0][0])


class TestCropHeight(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((4, 4, 4, 1), dtype = int)

    def test_4x4x4x1_3(self):
        result = ACDC.crop_height(self.image, 3)
        assert result.shape == (4, 3, 4, 1)
    
    def test_4x4x4x1_2(self):
        result = ACDC.crop_height(self.image, 2)
        assert result.shape == (4, 2, 4, 1)
    
    def test_mask_4x4x4_3(self):
        mask = np.zeros((4, 4, 4), dtype = int)
        result = ACDC.crop_height(mask, 3)
        assert result.shape == (4, 3, 4)


class TestCropWidth(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((4, 4, 4, 1), dtype = int)

    def test_4x4x4x1_3(self):
        result = ACDC.crop_width(self.image, 3)
        assert result.shape == (4, 4, 3, 1)
    
    def test_4x4x4x1_2(self):
        result = ACDC.crop_width(self.image, 2)
        assert result.shape == (4, 4, 2, 1)

    def test_mask_4x4x4_3(self):
        mask = np.zeros((4, 4, 4), dtype = int)
        result = ACDC.crop_width(mask, 3)
        assert result.shape == (4, 4, 3)


class TestPadHeight(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((2, 2, 2, 1), dtype = int)

    def test_2x2x2x1_3(self):
        result = ACDC.pad_height(self.image, 3)
        assert result.shape == (2, 3, 2, 1)
    
    def test_2x2x2x1_4(self):
        result = ACDC.pad_height(self.image, 4)
        assert result.shape == (2, 4, 2, 1)
    
    def test_mask_2x2x2_3(self):
        mask = np.zeros((2, 2, 2), dtype = int)
        result = ACDC.pad_height(mask, 3)
        assert result.shape == (2, 3, 2)


class TestPadWidth(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((2, 2, 2, 1), dtype = int)
    
    def test_2x2x2x1_3(self):
        result = ACDC.pad_width(self.image, 3)
        assert result.shape == (2, 2, 3, 1)
    
    def test_2x2x2x1_4(self):
        result = ACDC.pad_width(self.image, 4)
        assert result.shape == (2, 2, 4, 1)

    def test_mask_2x2x2_3(self):
        mask = np.zeros((2, 2, 2), dtype = int)
        result = ACDC.pad_width(mask, 3)
        assert result.shape == (2, 2, 3)


class TestNormalize(unittest.TestCase):
    def test_0111_012(self):
        slice1 = np.array([[0], [1], [1]])
        slice2 = np.array([[0], [1], [2]])
        generator = (slice1, slice2)
        result_generator = ACDC.normalize(generator)
        result = list(result_generator)
        assert (result[0] == np.array([[0], [255], [255]])).all()
        assert (result[1] == np.array([[0], [127], [255]])).all()


class TestCreateFrameFilenameImage(unittest.TestCase):
    def test_patient001_frame01(self):
        filename = ACDC.create_frame_filename_image("../../datasets_samples/ACDC/patient001", 1)
        assert filename == "patient001_frame01.nii.gz"        


class TestCreateFrameFilenameMask(unittest.TestCase):
    def test_patient001_frame12(self):
        filename = ACDC.create_frame_filename_mask("../../datasets_samples/ACDC/patient001", 12)
        assert filename == "patient001_frame12_gt.nii.gz"        


class TestGetFramesPaths(unittest.TestCase):
    def test_patient001_ED_image(self):
        result_generator = ACDC.get_frames_paths("../../datasets_samples/ACDC/patient001", "ED", ACDC.create_frame_filename_image)
        result = list(result_generator)
        assert result[0] == "../../datasets_samples/ACDC/patient001/patient001_frame01.nii.gz"    

    def test_patient001_ED_mask(self):
        result_generator = ACDC.get_frames_paths("../../datasets_samples/ACDC/patient001", "ED", ACDC.create_frame_filename_mask)
        result = list(result_generator)
        assert result[0] == "../../datasets_samples/ACDC/patient001/patient001_frame01_gt.nii.gz" 

    def test_patient001_ES_image(self):
        result_generator = ACDC.get_frames_paths("../../datasets_samples/ACDC/patient001", "ES", ACDC.create_frame_filename_image)
        result = list(result_generator)
        assert result[0] == "../../datasets_samples/ACDC/patient001/patient001_frame12.nii.gz"    

    def test_patient001_both_image(self):
        result_generator = ACDC.get_frames_paths("../../datasets_samples/ACDC/patient001", "both", ACDC.create_frame_filename_image)
        result = list(result_generator)
        assert result[0] == "../../datasets_samples/ACDC/patient001/patient001_frame01.nii.gz"    
        assert result[1] == "../../datasets_samples/ACDC/patient001/patient001_frame12.nii.gz"    

    def test_incorrect_directory(self):
        with self.assertRaises(EnvironmentError):
            result_generator = ACDC.get_frames_paths("../../datasets_samples/ACDC", "ED", ACDC.create_frame_filename_image)


class TestLoadNiftiImage(unittest.TestCase):
    def test_patient001(self):
        image = ACDC.load_nifti_image("../../datasets_samples/ACDC/patient001/patient001_frame01.nii.gz")
        assert type(image) == np.ndarray


class TestBinarizeMaskIfOneStructure(unittest.TestCase):
    def setUp(self):
        self.mask = np.array([0, 1, 2, 3])
     
    def test_RV(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "RV")
        assert (result == np.array([0, 1, 0, 0])).all()   

    def test_MYO(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "MYO")
        assert (result == np.array([0, 0, 1, 0])).all()   

    def test_LV(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "LV")
        assert (result == np.array([0, 0, 0, 1])).all()

    def test_all(self):
        result = ACDC.binarize_mask_if_one_structure(self.mask, "all")
        assert (result == self.mask).all()

