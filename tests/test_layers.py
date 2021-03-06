#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
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

        l_in = InputLayer(num_units=2)
        test_data = np.zeros((3, 2))
        l_in.set_inputs(test_data)

        self.assertEqual(l_in.a.shape, (3, 2))
        self.assertEqual(l_in.num_units, 2)

    def test_cast_to_string(self):

        l_in = InputLayer(name="Input", num_units=2)

        self.assertEqual(str(l_in),
                         'Input: 2')


class TestDenseLayer(unittest.TestCase):

    def setUp(self):

        self.l_in = InputLayer(num_units=2)

    def test_incoming_layer(self):

        l_hidden = DenseLayer(input_layer=self.l_in, num_units=10)

        self.assertEqual(l_hidden.input_layer, self.l_in)

    def test_num_units(self):

        l_hidden = DenseLayer(input_layer=self.l_in, num_units=10)

        self.assertEqual(l_hidden.num_units, 10)

    def test_weights_shape(self):

        l_hidden = DenseLayer(input_layer=self.l_in, num_units=2)
        l_hidden.initialise_weights()

        # Add 1 for baises
        self.assertEqual(l_hidden.W.shape, (3, 2))

    def test_weights_values(self):

        l_in = InputLayer(num_units=100)
        l_hidden = DenseLayer(input_layer=l_in, num_units=700)
        l_hidden.initialise_weights()

        self.assertTrue(np.max(l_hidden.W) <= 0.1)  # 0.5 is not a HARD rule
        self.assertTrue(np.min(l_hidden.W) >= -0.1)

    @unittest.skip("Biases now included in weights matrix")
    def test_bias_units_shape(self):

        l_hidden = DenseLayer(input_layer=self.l_in, num_units=10)

        self.assertEqual(l_hidden.b.shape, (1, 10))

    def test_activation(self):
        """Test the correct computation of activations using a
        simple perceptron with known weights"""

        l_in = InputLayer(num_units=2)
        l_in.set_inputs(np.array([[0, 1]]))

        l_output = DenseLayer(input_layer=l_in, num_units=2)

        l_output.W = np.array([
            [0.5, 0.5],
            [0.1, 0.2],
            [0.3, 0.4],
            ])

        # h = [[0.5 * 1 + 0.1 * 0 + 0.3 * 1, 0.5 * 1 + 0.2 * 0 + 0.4 * 1]]
        np.testing.assert_almost_equal(l_output.h_x(),
                                       np.array([[0.8, 0.9]]),
                                       decimal=1,
                                       )
        # a = s(h)
        np.testing.assert_almost_equal(l_output.a_h(),
                                       np.array([[0.69, 0.71]]),
                                       decimal=1,
                                       )

    def test_cast_to_string(self):

        l_in = InputLayer(num_units=2)
        l_output = DenseLayer(name="Hidden Layer",
                              input_layer=l_in,
                              num_units=1)

        self.assertEqual(str(l_output),
                         "Hidden Layer: 1 [linearity: Sigmoid]")

    def test_dropout_activations(self):
        l_in = InputLayer(num_units=2)
        l_in.set_inputs(np.array([
            [0, 1],
            [1, 1],
            [0, 1],
            ]))

        l_output = DenseLayer(input_layer=l_in, 
                              num_units=2, 
                              dropout=0.5)

        l_output.a_h()
        self.assertTrue(len(np.where(l_output.a == 0)[0]) == 3)


    def test_dropout_string(self):

        l_in = InputLayer(num_units=2)
        l_output = DenseLayer(input_layer=l_in, 
                              num_units=2, 
                              dropout=0.5)

        self.assertEqual(str(l_output),
                         "Dense Layer: 2 [linearity: Sigmoid dropout: 50%]")
                        


class TestNetworkLayerOrder(unittest.TestCase):

    def setUp(self):
        self.l_in = InputLayer(num_units=2, name="l_in")
        self.l_hidden = DenseLayer(input_layer=self.l_in,
                                   num_units=10,
                                   linearity=Sigmoid,
                                   name="l_hidden",
                                   )
        self.l_output = DenseLayer(input_layer=self.l_hidden,
                                   num_units=7,
                                   linearity=Sigmoid,
                                   name="l_output",
                                   )

    def test_network_iterable(self):

        layers = []
        for layer in iterlayers(self.l_output):
            layers.append(layer)

        self.assertEqual(layers, [self.l_hidden, self.l_output])
