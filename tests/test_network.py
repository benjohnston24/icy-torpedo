#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.network import networkBase
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.linearities import Sigmoid
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 September  10:44:02 AEST 2016'
__license__ = 'MPL v2.0'


class network(networkBase):

    def network_defn(self):

        self.l_in = InputLayer(inputs=np.array([[1, 0]]), name="Input")
        self.l_hidden = DenseLayer(input_layer=self.l_in, hidden_units=2, name="Hidden")
        self.l_output = DenseLayer(input_layer=self.l_hidden, hidden_units=1, name="Output")

        # Construct known weights
        self.l_hidden.W = np.array([
            [0.1, 0.4],
            [0.2, 0.5],])

        self.l_hidden.b = np.array([0.3, 0.6])

        self.l_output.W = np.array([
            [0.7],
            [0.8]])

        self.l_output.b = np.array([0.9])

        self.network = self.l_output


s = Sigmoid()

class TestNetwork(unittest.TestCase):

    def setUp(self):
        # Target output of network 
        self.target_output = 2.0944 

        self.net = network(targets=self.target_output)
        self.net.network_defn()

        # Expected output after forward propr 1.904
        self.expected_output = np.array([[
            (s(0.4) * 0.7) + (s(1) * 0.8) + 0.9]])

 
    def test_forward_and_back_prop(self):

        # Execute forward prop
        self.net.forwardprop()

        # Check the outputs are correct
        # Check hidden layer
        np.testing.assert_equal(self.net.l_hidden.h, np.array([[0.4, 1]]))
        np.testing.assert_allclose(self.net.l_hidden.a, np.array([[s(0.4), s(1)]]))

        # Check output layer
        np.testing.assert_equal(self.net.l_output.h, self.expected_output)
        np.testing.assert_equal(self.net.l_output.a, np.array(s(self.expected_output)))

        # Check back prop
        self.net.backprop()

        # Check output layer
        # delta = (s(1.904) - 2.0944) * s(1.904) * (1 - s(1.904)) 
        # dc_db = delta
        np.testing.assert_approx_equal(self.net.l_output.delta, np.array([-0.138133]),
                                       significant=4)
        np.testing.assert_approx_equal(self.net.l_output.dc_db, np.array([-0.138133]),
                                       significant=4)
 
        a_1 = np.array([[s(0.4), s(1)]])
        expected_result = np.dot(a_1.T, -0.138133)
        np.testing.assert_almost_equal(self.net.l_output.dc_dw, 
                                       expected_result,
                                       decimal=4,
                                       )

        # Check hidden layer
        np.testing.assert_almost_equal(self.net.l_hidden.delta,
                                       np.array([[-0.02323156, -0.02172688]]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.net.l_hidden.dc_db,
                                       np.array([[-0.02323156, -0.02172688]]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.net.l_hidden.dc_dw,
                                       np.array([[-0.02323156, -0.02172688], [0,0]]),
                                       decimal=4,
                                       )
