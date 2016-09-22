#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Check network can memorise single mnist image"""

# Imports
from icyTorpedo.resources import load_mnist_train_images, \
        load_mnist_train_labels, split_training_data
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.network import baseNetwork
from icyTorpedo.learningrates import FixedRate
from icyTorpedo.linearities import Linear, Tanh
import numpy as np

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 22 September  09:41:56 AEST 2016'
__license__ = 'MPL v2.0'

def _main(*args, **kwargs):

    train_images = load_mnist_train_images()
    train_images = train_images.reshape((-1, 28 ** 2))
    train_labels = load_mnist_train_labels()

    if False:
        num_samples=1000
        x_train = train_images[:num_samples,:]
        x_train = x_train.reshape((num_samples, -1))
        y_train = train_labels[:num_samples,:]
        y_train = y_train.reshape((num_samples, -1))
        x_valid = x_train  # Repeat using training image to ensure network stops
        y_valid = y_train

    else:
        x_train, y_train, x_valid, y_valid = split_training_data(train_images,
                                                                 train_labels,
                                                                 )

    l_input = InputLayer(num_units=28 ** 2, name="Input")
    l_hidden = DenseLayer(input_layer=l_input, 
                          num_units=700, 
                          name="Hidden")
    #l_hidden = DenseLayer(input_layer=l_input, num_units=(28**2), name="Hidden")
    l_output = DenseLayer(input_layer=l_hidden, 
                          num_units=10, 
                          linearity=Linear,
                          name="Output")

    net = baseNetwork(
            network_layers=[l_input, l_hidden, l_output],
            train_data=(x_train, y_train),
            valid_data=(x_valid, y_valid),
            eta=FixedRate(0.00001),
            max_epochs=1000,
            verbose=True,
            log_data=False,
            )

    # Log the descriptor
    net.log(str(net))

    net.train() 


if __name__ == "__main__":
    _main()
