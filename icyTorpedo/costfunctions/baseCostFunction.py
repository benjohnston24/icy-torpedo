#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import numpy as np

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  21:36:48 AEST 2016'
__license__ = 'MPL v2.0'


class baseCostFunction(object):

    def __init__(self, name="baseCostFunction", *args, **kwargs):
        self.name = name

    def __call__(self):
        """Override this method with the evaluation of the cost function"""
        pass  # pragma: no cover

    def prime(self):
        """Override this method with the derviative of the cost function"""
        pass  # pragma: no cover

    def __str__(self):

        return self.name
