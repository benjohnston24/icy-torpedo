#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Linear"""

# Imports
import numpy as np
from .baseLinearity import baseLinearity

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Sunday 18 September  23:54:39 AEST 2016'
__license__ = 'MPL v2.0'


class Linear(baseLinearity):
    """Linear function class"""

    def __init__(self, name="Linear", pseudo_inverse=True, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)

        self.name = name
        self.pseudo_inverse = pseudo_inverse

    @classmethod
    def __call__(cls, z):
        """Compute the linear function

        g(z) = z
        """

        return z

    @classmethod
    def prime(cls, z):
        """Compute the derivative of the linear

        g'(z) = 1
        """

        return np.ones(z.shape)
