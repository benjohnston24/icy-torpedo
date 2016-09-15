#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
from .baseCostFunction import baseCostFunction
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  21:59:14 AEST 2016'
__license__ = 'MPL v2.0'


class SquaredError(baseCostFunction):

    def __init__(self, *args, **kwargs):
        super(SquaredError, self).__init__(*args, **kwargs)

    def __call__(self, output, target):
        """Compute the squared error

        sqe = 0.5 * ((target - output) ** 2)
        """

        return 0.5 * ((target - output) ** 2)

    def prime(self, output, target):
        """Compute the derivative of the squared error

        sqe_prime = output - target
        """

        return output - target
