#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Test the resources module of the package
"""

# Imports

import unittest
from unittest.mock import mock_open
import icyTorpedo.resources as resources
from icyTorpedo.resources import load_mnist_test_images, load_mnist_test_labels, \
        load_mnist_train_images, load_mnist_train_labels, KAGGLE_LANDMARKS, \
        load_prepared_indices, load_from_kaggle_by_index
import os
import pandas
import numpy as np
import random

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  20:43:54 AEST 2016'
__license__ = 'MPL v2.0'

mock_file_open = mock_open()


def assert_data_division(test_obj, x_train, y_train, x_valid, y_valid, split_ratio, split_ratio_calculated):
    # Check equal lengths
    test_obj.assertEqual(len(x_train), len(y_train), 'x and y train dataset lengths not equal: %d != %d' %
                          (len(x_train), len(y_train)))
    test_obj.assertEqual(len(x_valid), len(y_valid), 'x and y valid dataset lengths not equal: %d != %d' %
                          (len(x_valid), len(y_valid)))
    # Check the correct ratios
    test_obj.assertEqual(split_ratio_calculated, split_ratio,
                          'incorrect split ratio: %0.2f' % split_ratio_calculated)


class TestResources(unittest.TestCase):
    """Test the resources"""

    def setUp(self):
        self.train_data_extract_landmarks = pandas.DataFrame({
            'left_eye_center_x': pandas.Series([1, 1]),
            'left_eye_center_y': pandas.Series([2, 2]),
            'left_eye_inner_corner_x': pandas.Series([3, 3]),
            'right_eye_center_x': pandas.Series([4, 4]),
            'right_eye_center_y': pandas.Series([5, 5]),
            'Image': pandas.Series(["255 255 255 255", "255 255 255 255"]),
        })

        self.y = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]]).T

        self.y_max = np.max(self.y)
        self.y = self.y / self.y_max 
        self.y_mean = np.mean(self.y)
        self.y = self.y - self.y_mean 

    def test_resources_path(self):
        """Test the correct resources path """
        # Check the path is correct
        self.assertEqual(os.path.relpath(resources.RESOURCE_DIR, __file__),
                         '../../icyTorpedo/resources')

    def test_training_set_filename(self):
        """Test the training set filename"""
        # Check the default training set name
        self.assertEqual(os.path.basename(resources.DEFAULT_TRAIN_SET), 'training.csv')

    def test_load_training_data(self):
        """Load the training set"""
        train_data = resources.load_training_data()
        # Check the default number of training samples
        self.assertEqual(train_data.shape[0], 7049, 'incorrect number of training samples %d != %d' %
                         (train_data.shape[0], 7049))
        self.assertEqual(train_data.shape[1], 31, 'incorrect number of training features %d != %d' %
                         (train_data.shape[1], 31))

    def test_load_data(self):
        """Load the data set with landmarks extracted and training / validation sets split"""
        train_data, offsets = resources.load_data()
        self.assertEqual(len(train_data), 4)
        self.assertEqual(train_data[0].shape[0], train_data[1].shape[0])
        self.assertEqual(train_data[2].shape[0], train_data[3].shape[0])
        self.assertEqual(train_data[0].shape[1], train_data[2].shape[1])
        self.assertEqual(train_data[1].shape[1], train_data[3].shape[1])

    def test_zero_mean(self):
        """Test the loaded data has zero mean"""
        train_data = resources.load_training_data()
        train_data = resources.remove_incomplete_data(train_data)
        x, y, _ = resources.extract_image_landmarks(train_data)

        np.testing.assert_almost_equal(np.mean(x), 0, decimal=4)
        np.testing.assert_almost_equal(np.mean(y), 0, decimal=4)

    def test_max_min_training_data(self):
        """Test the max and min of the data"""
        train_data = resources.load_training_data()
        train_data = resources.remove_incomplete_data(train_data)
        x, y, _ = resources.extract_image_landmarks(train_data)

        self.assertTrue(np.max(x) <= 1)
        self.assertTrue(np.min(x) >= -1)
        self.assertTrue(np.max(y) <= 1)
        self.assertTrue(np.min(y) >= -1)

    def test_load_data_different(self):
        """Test the loaded data is in fact different"""
        data, offsets = resources.load_data()
        x_train, y_train, x_valid, y_valid = data
        self.assertFalse(np.all(x_train == x_valid))
        self.assertFalse(np.all(y_train == y_valid))

    def test_load_data_from_different_file(self):
        """Test load_data tries to load from a different file, when not present and exception is raised"""

        with self.assertRaises(OSError):
            resources.load_data("new_training_set.csv")

    def test_remove_incomplete(self):
        """Remove incomplete data"""
        train_data = pandas.DataFrame(np.array([
                [1, 2],
                [3, 4],
                [5, np.NaN]]))
        selected_data = resources.remove_incomplete_data(train_data)
        self.assertLess(selected_data.shape[0], train_data.shape[0])
        self.assertEqual(selected_data.shape[1], 2)

    def test_image_landmark_extraction_shape(self):
        """Extract landmarks and images"""
        train_data = self.train_data_extract_landmarks
        x, y, _ = resources.extract_image_landmarks(train_data)
        self.assertEqual(len(x), len(y))
        self.assertEqual(x.shape[1], 4)
        self.assertEqual(y.shape[1], 5)
        
    def test_correct_offsets(self):
        """Test the offsets are correct"""
        train_data = self.train_data_extract_landmarks
        x, y, offsets = resources.extract_image_landmarks(train_data)
        np.testing.assert_allclose(offsets[2:], [self.y_max, self.y_mean])


    def test_image_landmark_extraction_x(self):
        """Test image extraction of extract_image_landmarks"""
        train_data = self.train_data_extract_landmarks
        x, y, _ = resources.extract_image_landmarks(train_data)
        np.testing.assert_allclose(x[0], [0, 0, 0, 0])

    def test_image_landmark_extraction_y_0(self):
        """Test landmark extraction of extract_image_landmarks 0"""
        train_data = self.train_data_extract_landmarks
        x, y, _ = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[0, 0], self.y[0, 0])

    def test_image_landmark_extraction_y_1(self):
        """Test landmark extraction of extract_image_landmarks 1"""
        train_data = self.train_data_extract_landmarks
        x, y, _ = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[0, 1], self.y[0, 1])

    def test_image_landmark_extraction_y_2(self):
        """Test landmark extraction of extract_image_landmarks 2"""
        train_data = self.train_data_extract_landmarks
        x, y, _ = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[1, 0], self.y[1, 0])

    def test_image_landmark_extraction_y_3(self):
        """Test landmark extraction of extract_image_landmarks 3"""
        train_data = self.train_data_extract_landmarks
        x, y, _ = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[1, 1], self.y[1, 1])

    def test_splitting_training_data(self):
        """Test default train / valid set split"""
        train_data = pandas.DataFrame({
            'left_eye_center_x': pandas.Series(random.sample(range(1000), 10)),
            'left_eye_center_y': pandas.Series(random.sample(range(1000), 10)),
            'left_eye_inner_corner_x': pandas.Series(random.sample(range(1000), 10)),
            'right_eye_center_x': pandas.Series(random.sample(range(1000), 10)),
            'right_eye_center_y': pandas.Series(random.sample(range(1000), 10)),
            'Image': pandas.Series([
               "".join(["%s " % str(x) for x in random.sample(range(255), 4)]).strip()
               for i in range(10)
                ]),
        })

        # Generate random images
        x, y, _ = resources.extract_image_landmarks(train_data)

        for split_ratio in [0.5, 0.7]:
            x_train, y_train, x_valid, y_valid = \
                    resources.split_training_data(x, y, split_ratio=split_ratio)
            split_ratio_calculated = np.round(len(x_train) / (len(x_train) + len(x_valid)), 1)
            with self.subTest(split_ratio=split_ratio):
                assert_data_division(self, x_train, y_train, x_valid, y_valid, split_ratio, split_ratio_calculated)
                # Check the shape of the features
                self.assertEqual(x_train.shape[1], 4)
                self.assertEqual(y_train.shape[1], 5)
                self.assertEqual(x_train.shape[0], y_train.shape[0])
                self.assertEqual(x_valid.shape[1], 4)
                self.assertEqual(y_valid.shape[1], 5)
                self.assertEqual(x_valid.shape[0], y_valid.shape[0])

                # Check the data is not equal
                self.assertFalse(np.all(x_train == x_valid))
                self.assertFalse(np.all(y_train == y_valid))


class TestMNISTData(unittest.TestCase):

    def test_load_mnist_test_images(self):
        """Test the MNIST test set images load correctly"""

        images = load_mnist_test_images()

        self.assertEqual(images.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(images.shape[0], 10000,
                         "The test set should contain 10k images")
        self.assertEqual(images.shape[1], 28,
                         "Each test set image should be 28 x 28 pixels")
        self.assertEqual(images.shape[2], 28,
                         "Each test set image should be 28 x 28 pixels")
        self.assertTrue(np.amax(images) <= 1,
                        "Image values must be less than 1")
        self.assertTrue(np.amin(images) >= -1,
                        "Image values must be less than 0")
        np.testing.assert_almost_equal(np.mean(images), 0,
                                       decimal=5,
                                       err_msg="Mean not equal to 0")

    def test_load_mnist_test_labels(self):
        """Test the MNIST test set labels load correctly"""

        first_label = np.zeros((10))
        first_label[7] = 1

        last_label = np.zeros((10))
        last_label[6] = 1

        labels = load_mnist_test_labels()

        self.assertEqual(labels.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(labels.shape[0], 10000,
                         "The test set should contain 10k images")
        self.assertEqual(labels.shape[1], 10,
                         "The labels should be present as a vector of length 10")
        np.testing.assert_equal(labels[0], first_label,
                                "First sample incorrectly labelled")
        np.testing.assert_equal(labels[-1], last_label,
                                "First sample incorrectly labelled")

    def test_load_mnist_train_images(self):
        """Test the MNIST training set images load correctly"""
        images = load_mnist_train_images()

        self.assertEqual(images.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(images.shape[0], 60000,
                         "The test set should contain 10k images")
        self.assertEqual(images.shape[1], 28,
                         "Each test set image should be 28 x 28 pixels")
        self.assertEqual(images.shape[2], 28,
                         "Each test set image should be 28 x 28 pixels")
        self.assertTrue(np.max(images) <= 1,
                        "Image values must be less than 1")
        self.assertTrue(np.min(images) >= -1,
                        "Image values must be less than 0")
        np.testing.assert_almost_equal(np.mean(images), 0,
                                       decimal=5,
                                       err_msg="Mean not equal to 0")

    def test_load_mnist_training_labels(self):
        """Test the MNIST test set labels load correctly"""

        first_label = np.zeros((10))
        first_label[5] = 1

        last_label = np.zeros((10))
        last_label[8] = 1

        labels = load_mnist_train_labels()

        self.assertEqual(labels.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(labels.shape[0], 60000,
                         "The test set should contain 10k images")
        self.assertEqual(labels.shape[1], 10,
                         "The labels should be present as a vector of length 10")
        np.testing.assert_equal(labels[0], first_label,
                                "First sample incorrectly labelled")
        np.testing.assert_equal(labels[-1], last_label,
                                "First sample incorrectly labelled")

class TestSpecialisedLandmarks(unittest.TestCase):

    @unittest.skip("Run only when new data is generated, a very long test")
    def test_pickled_specialised_landmarks(self):
        """Test the validity of the pickled specialist landmark data"""

        for idx, col in enumerate(KAGGLE_LANDMARKS):
            with self.subTest(idx=idx):
                train_idx, valid_idx, test_idx = load_prepared_indices(idx) 
                data = load_from_kaggle_by_index(index=train_idx.tolist(),
                                                          cols=col)
                x_train, y_train = data[:2]
                data = load_from_kaggle_by_index(index=valid_idx.tolist(),
                                                             cols=col)
                x_valid, y_valid = data[:2]
                self.assertFalse(np.any(y_train is np.nan))
                self.assertFalse(np.any(y_valid is np.nan))
                self.assertFalse(np.any(x_train > 1))
                self.assertFalse(np.any(y_train > 1))
                self.assertFalse(np.any(x_train < -1))
                self.assertFalse(np.any(y_train < -1))

        # Just load the data, if an assertion error is raised the data is corrupt

    def test_all_test_indices_identical(self):
        """Test the saved test indices are identical"""

        test_indices = []
        for idx, col in enumerate(KAGGLE_LANDMARKS):
            train_idx, valid_idx, test_idx = load_prepared_indices(idx) 
            test_indices.append(test_idx)

        for idx, col in enumerate(KAGGLE_LANDMARKS):
            with self.subTest(idx=idx):
                self.assertTrue(len(test_indices[idx]) == 0)  # Use test.csv as the kaggle test set
                # self.assertTrue(np.all(test_indices[0] == test_indices[idx]))
