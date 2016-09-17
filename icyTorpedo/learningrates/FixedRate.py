#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 September  14:19:54 AEST 2016'
__license__ = 'MPL v2.0'


class FixedRate(object):
    """Fixed learning rate

    Parameters
    ----------

    value: The value of the learning rate

    Usage
    ---------

    To obtain the learning rate value simple call the object e.g.

    eta = FixedRate(0.01)

    learn_rate = eta()

    assert(lean_rate == 0.01)

    """

    def __init__(self, value, name="FixedRate", *args, **kwargs):
        self.name = name
        self.value = value

    def __call__(self):
        return self.value

    def __str__(self):

        return "%s: %E" % (self.name, self.value)
