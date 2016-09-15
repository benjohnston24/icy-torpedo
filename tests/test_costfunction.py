#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.costfunctions import SquaredError 
from icyTorpedo.costfunctions.baseCostFunction import baseCostFunction
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  21:46:19 AEST 2016'
__license__ = 'MPL v2.0'


class TestBaseCostFunctions(unittest.TestCase):

    def test_name(self):
        c = baseCostFunction(name='C')

        self.assertEqual(c.name, 'C')


class TestSquaredError(unittest.TestCase):

    def test_squared_error_name(self):

        c = SquaredError('C')

        self.assertEqual(c.name, 'C')

    def test_squared_error_1(self):

        c = SquaredError()

        self.assertEqual(c(4, 2), 2)

    def test_squared_error_2(self):

        c = SquaredError()

        self.assertEqual(c(-4, -2), 2)

    def test_squared_error_3(self):

        c = SquaredError()

        self.assertEqual(c(output=4, target=-2), 18)

    def test_squared_error_4(self):

        c = SquaredError()

        self.assertEqual(c(output=-2, target=4), 18)

    def test_squared_prime_1(self):

        c = SquaredError()

        self.assertEqual(c.prime(output=-2, target=4), -6)

    def test_squared_prime_2(self):

        c = SquaredError()

        self.assertEqual(c.prime(output=2, target=4), -2)

    def test_squared_prime_3(self):

        c = SquaredError()

        self.assertEqual(c.prime(output=4, target=1), 3)
