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


class TestMnistSingle(unittest.TestCase):

    def setUp(self):
        self.l_in = InputLayer(num_units=28 ** 2, name="Input")
        self.l_hidden = DenseLayer(input_layer=self.l_in, hidden_units=2, name="Hidden")
        self.network = DenseLayer(input_layer=self.l_hidden, hidden_units=10, name="Output")


        # Load a single image
        self.image = load_mnist_train_images()[0]
        self.label = load_mnist_train_labels()[0]

        self.net = baseNetwork(
                network_layers =[self.l_in, self.l_hidden, self.network],
                eta=FixedRate(0.1),
                max_epochs=1000,
                )

        self.net.x_train = self.image.reshape((1, 28 ** 2))
        self.l_in.set_inputs(self.net.x_train)
        self.net.y_train = self.label.reshape((1, 10))
        self.net_x_valid = self.net.x_train
        self.net_y_valid = self.net.y_train

        self.net.targets = self.net.y_train

    def test_memorise_single_sample(self):

        #print("Correct value")
        #print("{:^10}{}".format("-", "".join(["%0.4f, " % x for x in self.net.y_train[0]])))
        #print("")
        self.net.train()

        predictions = self.net.output_layer.a

        np.testing.assert_almost_equal(predictions,
                np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]),
                decimal=1,
                )
