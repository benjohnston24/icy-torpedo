#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.linearities import Sigmoid
from icyTorpedo.linearities.baseLinearity import baseLinearity
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  21:11:15 AEST 2016'
__license__ = 'MPL v2.0'


class TestBaseActivation(unittest.TestCase):

    def test_name(self):
        g = baseLinearity(name='g')

        self.assertEqual(g.name, 'g')


class TestSigmoid(unittest.TestCase):

    def test_sigmoid_name(self):
        g = Sigmoid(name='g')

        self.assertEqual(g.name, 'g')

    def test_sigmoid_basic_call(self):
        g = Sigmoid()
        
        np.testing.assert_allclose(g(0), 0.5)

    def test_sigmoid_basic_call_max(self):
        g = Sigmoid()

        np.testing.assert_allclose(g(np.inf), 1)

    def test_sigmoid_basic_call_min(self):
        g = Sigmoid()

        np.testing.assert_allclose(g(-np.inf), 0)

    def test_sigmoid_deriv_call(self):
        g = Sigmoid()

        np.testing.assert_allclose(g.prime(0), 0.25)

    def test_sigmoid_deriv_call_min(self):
        g = Sigmoid()

        np.testing.assert_allclose(g.prime(np.inf), 0)

    def test_sigmoid_deriv_call_max(self):
        g = Sigmoid()

        np.testing.assert_allclose(g.prime(-np.inf), 0)
