#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Fixed momentum class"""

# Imports


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 13 October  10:35:22 AEDT 2016'
__license__ = 'MPL v2.0'


class FixedMomentum(object):
    """Fixed momentum

    Parameters
    ----------

    value: The value of the momentum [default 0.9]

    Usage
    ---------

    To obtain the momentum value simple call the object e.g.

    momentum = FixedMomentum(0.9)

    m = eta()

    assert(m == 0.9)

    """

    def __init__(self, value=0.9, name="FixedMomentum", *args, **kwargs):
        self.name = name
        self.value = value

    def __call__(self):
        return self.value

    def __str__(self):

        return "%s: %E" % (self.name, self.value)
