#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.layers.baseLayer import baseLayer
from icyTorpedo.layers import InputLayer, DenseLayer, iterlayers
from icyTorpedo.linearities import Sigmoid
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  22:49:12 AEST 2016'
__license__ = 'MPL v2.0'


class TestInputLayer(unittest.TestCase):

    def test_input_name(self):

        l_in = InputLayer(name="Input")

        self.assertEqual(l_in.name, "Input")

    def test_input_shape(self):

        test_data = np.ones((5,2))
        l_in = InputLayer(inputs=test_data)

        self.assertEqual(l_in.input_shape, 5)
        self.assertEqual(l_in.num_units, 2)
        self.assertTrue(np.all(l_in.a == test_data))


class TestDenseLayer(unittest.TestCase):

    def setUp(self):

        test_data = np.ones((5,2))
        self.l_in = InputLayer(inputs=test_data)

    def test_incoming_layer(self):

        l_hidden = DenseLayer(input_layer=self.l_in, hidden_units=10)

        self.assertEqual(l_hidden.input_layer, self.l_in)

    def test_num_hidden_units(self):

        l_hidden = DenseLayer(input_layer=self.l_in, hidden_units=10)

        self.assertEqual(l_hidden.num_units, 10)

    def test_weights_shape(self):

        l_hidden = DenseLayer(input_layer=self.l_in, hidden_units=10)

        self.assertEqual(l_hidden.W.shape, (2,10))

    def test_bias_units_shape(self):

        l_hidden = DenseLayer(input_layer=self.l_in, hidden_units=10)

        self.assertEqual(l_hidden.b.shape, (10,))

    def test_activation(self):
        """Test the correct computation of activations using a simple perceptron with known weights"""

        l_in = InputLayer(np.array([[0,1]]))
        s = Sigmoid()

        l_output = DenseLayer(input_layer=l_in, hidden_units=1)

        l_output.W = np.array([[0.2],[0.1]])
        l_output.b = np.array([0.3])

        self.assertEqual(l_output.h_x(), 0.4)
        self.assertEqual(l_output.a_h(), s(0.4))


class TestNetworkLayerOrder(unittest.TestCase):

    def setUp(self):

        test_data = np.ones((5,2))
        self.l_in = InputLayer(inputs=test_data, name="l_in")
        self.l_hidden = DenseLayer(input_layer=self.l_in,
                                   hidden_units=10,
                                   linearity=Sigmoid,
                                   name="l_hidden",
                                   )
        self.l_output = DenseLayer(input_layer=self.l_hidden,
                                   hidden_units=7,
                                   linearity=Sigmoid,
                                   name="l_output",
                                   )

    def test_network_iterable(self):

        layers = []
        for layer in iterlayers(self.l_output):
            layers.append(layer)

        self.assertEqual(layers, [self.l_hidden, self.l_output])
