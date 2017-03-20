#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from unittest.mock import MagicMock, patch
import os
from icyTorpedo.resources import load_cifar10_train_data, \
        load_cifar10_test_data
from copy import deepcopy as copy
import numpy as np

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Monday 28 November  12:00:24 AEDT 2016'
__license__ = 'MPL v2.0'


class TestCIFAR10(unittest.TestCase):

    def setUp(self):
        pass

    def test_load_train_data(self):
        labels, images = load_cifar10_train_data()

        self.assertEqual(labels.shape,(5000,1))
        self.assertTrue(isinstance(labels[0], ing))
        self.assertEqual(images.shape,(5000,32,32,3))

    def test_load_test_data(self):
        labels, images = load_cifar10_test_data()

        self.assertEqual(labels.shape,(1000,1))
        self.assertTrue(isinstance(labels[0], ing))
        self.assertEqual(images.shape,(1000,32,32,3))

