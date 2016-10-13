#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Test the muct data set"""

# Imports
import unittest
from unittest.mock import MagicMock, patch
import os
from icyTorpedo.resources import MUCTData
from copy import deepcopy as copy
import numpy as np

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Monday 10 October  22:44:27 AEDT 2016'
__license__ = 'MPL v2.0'

INITIAL_IMAGE_FOLDER = copy(MUCTData.get_muct_image_folder())

mock_image_list = MagicMock(return_value=['i000qa-fn.jpg', 'i000qa-fn.jpg','i000qa-fn.jpg'])


class TestMUCT(unittest.TestCase):

    def setUp(self):
        pass

    def test_folder_location(self):
        """Test the correct change of image folder location"""
        MUCTData.set_muct_image_folder('test_folder')

        self.assertEqual(MUCTData.get_muct_image_folder(),
                         'test_folder')

    def test_image_list_file_location(self):
        """Test the location of the list of images"""
        self.assertEqual(MUCTData.MUCT_FRONTAL_IMAGE_LIST,
                '/home/ben/Workspace/icy-torpedo/icyTorpedo/'\
                'resources/muct_frontal_images.txt')

    def test_yield_image_list(self):
        """Test the image list yields the correct locations"""
        MUCTData.set_muct_image_folder(INITIAL_IMAGE_FOLDER)
        counter = 0
        for image in MUCTData.yield_frontal_images_list():
            filename, ext = os.path.splitext(image)
            self.assertEqual(ext, '.jpg')
            counter += 1

        # Test the correct number of images
        self.assertEqual(counter, 2253)

    @patch('icyTorpedo.resources.MUCTData.yield_frontal_images_list', mock_image_list)
    def test_read_frontal_image_names(self):
        """Test reading frontal image names"""

        MUCTData.set_muct_image_folder(INITIAL_IMAGE_FOLDER)
        image = next(MUCTData.yield_frontal_images_names())
        self.assertEqual(image,'i000qa-fn')


    def test_landmark_file_location(self):
        """Test the location of the landmarks file"""
        self.assertEqual(MUCTData.MUCT_IMAGE_LANDMARKS,
                '/home/ben/Workspace/icy-torpedo/icyTorpedo/'\
                'resources/me17-opencv.csv')

    def test_landmarks_from_list(self):
        """Test the correct landmarks returned"""

        landmarks = MUCTData.read_landmarks_from_list()
        landmarks_names = landmarks.name.values
        for image in MUCTData.yield_frontal_images_names():
            with self.subTest(image=image):
                self.assertTrue(image in landmarks_names) 

        self.assertEqual(landmarks.shape, (2253, 35))

    def test_read_image(self):
        """Test the reading of an image"""
        MUCTData.set_muct_image_folder(INITIAL_IMAGE_FOLDER)
        img = MUCTData.read_image('i000qa-fn.jpg')

        self.assertEqual(img.shape, (640, 480))

    @patch('icyTorpedo.resources.MUCTData.yield_frontal_images_list', mock_image_list)
    def test_read_image_list(self):
        """Test reading a few images"""

        images = MUCTData.read_images_from_list()
        self.assertEqual(images.shape, (3, 640 * 480))

    def test_split_test_validation_train_sets(self):
        """Test the correct construction of the test, train, validation sets"""

        train_images, valid_images, test_images = MUCTData.test_train_split_names(
                split_ratio_test = 0.1, split_ratio_train = 0.7)

        test_set_length = 226 
        train_set_length = 1418 
        valid_set_length = 2253 - test_set_length - train_set_length 

        self.assertEqual(len(test_images), test_set_length)
        self.assertEqual(len(train_images), train_set_length)
        self.assertEqual(len(valid_images), valid_set_length)

