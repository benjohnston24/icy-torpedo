#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from icyTorpedo.learningrates import FixedRate


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 September  14:23:29 AEST 2016'
__license__ = 'MPL v2.0'


class TestFixedRate(unittest.TestCase):

    def test_return_value(self):

        eta = FixedRate(0.01)

        self.assertEqual(eta(), 0.01)
        self.assertEqual(str(eta), 'FixedRate: 1.000000E-02')
