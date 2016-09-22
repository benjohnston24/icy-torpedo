#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Tanh"""

# Imports
import numpy as np
from .baseLinearity import baseLinearity

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 22 September  13:34:48 AEST 2016'
__license__ = 'MPL v2.0'


class Tanh(baseLinearity):
    """Tanh function class"""

    def __init__(self, name="Tanh", *args, **kwargs):
        super(Tanh, self).__init__(*args, **kwargs)

        self.name = name

    @classmethod
    def __call__(cls, z):
        """Compute the linear function
        
        g(z) = tanh(z) 
        """

        return np.tanh(z)

    @classmethod
    def prime(cls, z):
        """Compute the derivative of the linear 

        g'(z) = 1 
        """
        return 1 - (cls.__call__(z) ** 2  )
