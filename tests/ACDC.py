import numpy as np
import logging
import unittest

from medreaders import ACDC

logging.basicConfig(level = logging.INFO)

"""
class TestLoad(unittest.TestCase):
    def test_RV_ED(self):
        ACDC.load("datasets_samples/ACDC", "RV", "ED")


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
"""

class TestNormalize(unittest.TestCase):
    def test_0112_2110(self):
        slice1 = np.array([0, 1, 1])
        slice2 = np.array([0, 1, 2])
        generator = (slice1, slice2)
        result_generator = ACDC.normalize(generator)
        result = list(result_generator)
        assert (result[0] == np.array([0, 255, 255])).all()
        assert (result[1] == np.array([0, 127, 255])).all()
        

