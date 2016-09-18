#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
import sys
from icyTorpedo.mnist.runMnist import _options, _main


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Sunday 18 September  22:30:11 AEST 2016'
__license__ = 'MPL v2.0'


class TestRunMnist(unittest.TestCase):

    def test_options_defaults(self):

        sys.argv = ['']

        args = _options()

        self.assertEqual(args.desc, '')
        self.assertEqual(args.name, 'mnist')
        self.assertEqual(args.nodes, 700)
        self.assertEqual(args.epochs, 1000)

    def test_options_specifics(self):

        sys.argv[1:] = ['-d', 'mnist description',
                        '-s', 'mnist_name',
                        '-n', '10',
                        '-m', '12',
                        ]

        args = _options()

        self.assertEqual(args.desc, 'mnist description')
        self.assertEqual(args.name, 'mnist_name')
        self.assertEqual(args.nodes, 10)
        self.assertEqual(args.epochs, 12)
