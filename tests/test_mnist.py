#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.resources import load_mnist_train_images, \
        load_mnist_train_labels
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.network import baseNetwork
from icyTorpedo.learningrates import FixedRate
import numpy as np

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 September  22:41:04 AEST 2016'
__license__ = 'MPL v2.0'


class TestMnistSingleSample(unittest.TestCase):

    def setUp(self):
        self.l_in = InputLayer(num_units=28 ** 2, name="Input")
        self.l_hidden = DenseLayer(input_layer=self.l_in,
                                   num_units=2,
                                   name="Hidden")
        self.network = DenseLayer(input_layer=self.l_hidden,
                                  num_units=10,
                                  name="Output")

        # Load a single image
        self.image = load_mnist_train_images()[0]
        self.label = load_mnist_train_labels()[0]

        self.net = baseNetwork(
                network_layers=[self.l_in, self.l_hidden, self.network],
                eta=FixedRate(0.1),
                max_epochs=1000,
                log_data=False,
                verbose=False,
                )

        self.net.x_train = self.image.reshape((1, 28 ** 2))
        self.l_in.set_inputs(self.net.x_train)
        self.net.y_train = self.label.reshape((1, 10))
        self.net.x_valid = self.net.x_train
        self.net.y_valid = self.net.y_train

    def test_memorise_single_sample(self):

        train_err, valid_err, correct_class = self.net.train()

        predictions = self.net.output_layer.a

        np.testing.assert_almost_equal(predictions,
                                       np.array([[0, 0, 0, 0, 0, 1,
                                                  0, 0, 0, 0]]),
                                       decimal=1,
                                       )

        np.testing.assert_almost_equal(train_err, 0, decimal=1)

        np.testing.assert_almost_equal(valid_err, 0, decimal=1)

        self.assertEqual(correct_class, 1)


class TestMnistDoubleSample(unittest.TestCase):

    def setUp(self):
        self.l_in = InputLayer(num_units=28 ** 2,
                               name="Input")
        self.l_hidden = DenseLayer(input_layer=self.l_in,
                                   num_units=2,
                                   name="Hidden")
        self.network = DenseLayer(input_layer=self.l_hidden,
                                  num_units=10,
                                  name="Output")

        # Load a single image
        self.image = load_mnist_train_images()[:2, :]
        self.image = self.image.reshape((-1, 28 ** 2))
        self.label = load_mnist_train_labels()[:2, :]

        # Split into training and validation tests
        self.x_train = self.image
        y_train = self.label
        x_valid = self.x_train
        y_valid = y_train

        self.net = baseNetwork(
                network_layers=[self.l_in, self.l_hidden, self.network],
                train_data=(self.x_train, y_train),
                valid_data=(x_valid, y_valid),
                eta=FixedRate(0.1),
                max_epochs=1000,
                log_data=False,
                verbose=False,
                )

    def test_can_execute_two_samples(self):

        train_err, valid_err, correct_class = self.net.train()

        predictions = self.net.predict(self.x_train)

        # Just check the shape, not the output as we are not
        # trying hard to train the network
        self.assertEqual(predictions.shape, (2, 10))

        np.testing.assert_almost_equal(train_err, 0, decimal=0)

        np.testing.assert_almost_equal(valid_err, 0, decimal=0)
