#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.resources import BioIdData
import numpy as np

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Friday 14 October  16:29:38 AEDT 2016'
__license__ = 'MPL v2.0'


class TestBioIdData(unittest.TestCase):

    def setUp(self):
        self.names = BioIdData.load_names()
        self.images = BioIdData.load_images(self.names)
        self.landmarks = BioIdData.load_landmarks(self.names)

    def test_number_names(self):
        """Test the correct number of names"""

        self.assertEqual(len(self.names), 1521)

    def test_number_images(self):
        """Test the number of images in the dataset"""

        self.assertEqual(self.images.shape, (1521, 286, 384))

    def test_number_landmarks(self):
        """Test the number of landmarks in the dataset"""

        self.assertEqual(self.landmarks.shape, (1521, 20, 2))

    def test_min_max_scaled_image(self):
        """Test the scaled images are within the -1 and 1 bounds"""

        scaled_images, max_img, mean_img = BioIdData.scale_data(self.images)

        self.assertTrue(np.max(scaled_images) <= 1)
        self.assertTrue(np.min(scaled_images) >= -1)
        np.testing.assert_almost_equal(np.mean(scaled_images), 0, decimal=2)

    def test_min_max_scaled_landmarks(self):
        """Test the scaled landmarks are within the -1 and 1 bounds"""

        scaled_landmarks, max_land, mean_land = BioIdData.scale_data(self.landmarks)
        self.assertTrue(np.max(scaled_landmarks) <= 1)
        self.assertTrue(np.min(scaled_landmarks) >= -1)
        np.testing.assert_almost_equal(np.mean(scaled_landmarks), 0, decimal=2)
