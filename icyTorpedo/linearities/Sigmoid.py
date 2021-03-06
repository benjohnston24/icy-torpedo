#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Sigmoid Function"""

# Imports
import numpy as np
from .baseLinearity import baseLinearity

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  21:58:11 AEST 2016'
__license__ = 'MPL v2.0'


class Sigmoid(baseLinearity):
    """Sigmoid function activation class"""

    def __init__(self, name="Sigmoid", *args, **kwargs):
        super(Sigmoid, self).__init__(*args, **kwargs)

        self.name = name

    @classmethod
    def __call__(cls, z):
        """Compute the sigmoid function

        g(z) =        1
                ---------------
                          -z
                  1   +   e

        """

        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def prime(cls, z):
        """Compute the derivative of the sigmoid

        g'(z) = g(z)(1 - g(z))
        """

        return cls.__call__(z) * (1 - cls.__call__(z))
