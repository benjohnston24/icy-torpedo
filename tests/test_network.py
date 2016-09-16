#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.network import baseNetwork
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.linearities import Sigmoid
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 September  10:44:02 AEST 2016'
__license__ = 'MPL v2.0'


s = Sigmoid()

class TestNetwork(unittest.TestCase):

    def setUp(self):

        self.reset()

    def reset(self):
        # Define the architecture of the network
        self.l_in = InputLayer(num_units=2, name="Input")
        self.l_hidden = DenseLayer(input_layer=self.l_in, hidden_units=2, name="Hidden")
        self.output_layer = DenseLayer(input_layer=self.l_hidden, hidden_units=1, name="Output")

        # Set the initial input values
        self.l_in.set_inputs(np.array([[1, 0]]))

        # Construct known weights
        self.l_hidden.W = np.array([
            [0.1, 0.4],
            [0.2, 0.5],])

        self.l_hidden.b = np.array([[0.3, 0.6]])

        self.output_layer.W = np.array([
            [0.7],
            [0.8]])

        self.output_layer.b = np.array([[0.9]])

        # Target output of network 
        self.target_output = 2.0944 

        self.net = baseNetwork(
                network_layers = [self.l_in, self.l_hidden, self.output_layer],
                targets=self.target_output,
                )

        # Expected output after forward propr 1.904
        self.expected_output = np.array([[
            (s(0.4) * 0.7) + (s(1) * 0.8) + 0.9]])

 
    def test_forward_and_back_prop(self):

        self.reset()

        # Execute forward prop
        self.net.forwardprop()

        # Check the outputs are correct
        # Check hidden layer
        np.testing.assert_equal(self.l_hidden.h, np.array([[0.4, 1]]))
        np.testing.assert_allclose(self.l_hidden.a, np.array([[s(0.4), s(1)]]))

        # Check output layer
        np.testing.assert_equal(self.net.output_layer.h, self.expected_output)
        np.testing.assert_equal(self.net.output_layer.a, np.array(s(self.expected_output)))


    def test_backprop(self):

        self.reset()

        self.net.forwardprop()

        # Check back prop
        self.net.backprop()

        # Check output layer
        # delta = (s(1.904) - 2.0944) * s(1.904) * (1 - s(1.904)) 
        # dc_db = delta
        np.testing.assert_approx_equal(self.net.output_layer.delta, np.array([-0.138133]),
                                       significant=4)
        np.testing.assert_approx_equal(self.net.output_layer.dc_db, np.array([-0.138133]),
                                       significant=4)
 
        a_1 = np.array([[s(0.4), s(1)]])
        expected_result = np.dot(a_1.T, -0.138133)
        np.testing.assert_almost_equal(self.net.output_layer.dc_dw, 
                                       expected_result,
                                       decimal=4,
                                       )

        # Check hidden layer
        np.testing.assert_almost_equal(self.l_hidden.delta,
                                       np.array([[-0.02323156, -0.02172688]]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.dc_db,
                                       np.array([[-0.02323156, -0.02172688]]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.dc_dw,
                                       np.array([[-0.02323156, -0.02172688], [0,0]]),
                                       decimal=4,
                                       )

    def test_update_weights(self):

        self.reset()

        self.net.forwardprop()
        self.net.backprop()

        self.net.updateweights()

        # Check weights

        np.testing.assert_almost_equal(self.net.output_layer.W,
                                       np.array([
                                           [0.7000827],
                                           [0.8010098]]),
                                       decimal=3,
                                       )
        np.testing.assert_almost_equal(self.net.output_layer.b,
                                       np.array([
                                           [0.90138133]]),
                                       decimal=3,
                                       )

        np.testing.assert_almost_equal(self.l_hidden.W,
                                       np.array([
                                           [0.100232316, 0.4],
                                           [0.200217269, 0.5]]),
                                       decimal=3,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.b,
                                       np.array([
                                           [0.300232316, 0.600217269]]),
                                       decimal=3,
                                       )
