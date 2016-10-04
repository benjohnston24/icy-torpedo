#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Base Linearity class"""

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 14 September  21:03:24 AEST 2016'
__license__ = 'MPL v2.0'


class baseLinearity(object):

    def __init__(self, name="baseLinearity", *args, **kwargs):
        self.name = name

    def __call__(self):
        """Override this method with the evaluation of the activation"""
        pass  # pragma: no cover

    def prime(self):
        """Override this method with the derviative of the activation"""
        pass  # pragma: no cover
