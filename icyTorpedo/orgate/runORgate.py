#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
from icyTorpedo.network import baseNetwork
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.linearities import Sigmoid, Linear
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 22 September  09:27:16 AEST 2016'
__license__ = 'MPL v2.0'

l_input = InputLayer(num_units=2, name="Input")
l_hidden = DenseLayer(input_layer=l_input, num_units=2, name="Hidden")
l_output = DenseLayer(input_layer=l_hidden, num_units=1, linearity=Linear, name="Output")

x_train = np.array([
    [1, 0],
    [0, 0],
    [1, 1],
    [0, 1],
    ])

y_train = np.array([
    [1],
    [0],
    [0],
    [1],
    ])

x_valid = np.array([
    [1, 0],
    [0, 0],
    ])

y_valid = np.array([
    [1],
    [0],
    ])


net = baseNetwork(
        network_layers=[l_input, l_hidden, l_output],
        train_data=(x_train, y_train),
        valid_data=(x_valid, y_valid),
        max_epochs=10,
        verbose=True,
        log_data=False,
        )

net.log(str(net))

net.train()
