#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Check network can memorise single mnist image"""

# Imports
from icyTorpedo.resources import load_mnist_train_images, \
        load_mnist_train_labels, load_mnist_test_images, \
        load_mnist_test_labels
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.network import baseNetwork
from icyTorpedo.learningrates import FixedRate
from icyTorpedo.momentum import FixedMomentum 
from icyTorpedo.linearities import Linear, Sigmoid, Tanh

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 22 September  09:41:56 AEST 2016'
__license__ = 'MPL v2.0'


def _main(*args, **kwargs):

    train_images = load_mnist_train_images()

    train_images = train_images.reshape((-1, 28 ** 2))
    train_labels = load_mnist_train_labels()

    valid_images = load_mnist_test_images()

    valid_images = valid_images.reshape((-1, 28 ** 2))
    valid_labels = load_mnist_test_labels()



    if False:
        x_train, y_train, x_valid, y_valid = split_training_data(train_images,
                                                                 train_labels,
                                                                 )
    x_train = train_images
    x_valid = valid_images
    y_train = train_labels
    y_valid = valid_labels


    l_input = InputLayer(num_units=28 ** 2, name="Input")
    l_hidden = DenseLayer(input_layer=l_input,
                          num_units=784,
                          linearity=Tanh(),
                          name="Hidden")
    l_output = DenseLayer(input_layer=l_hidden,
                          num_units=10,
                          linearity=Linear(),
                          name="Output")

    net = baseNetwork(
            network_layers=[l_input, l_hidden, l_output],
            train_data=(x_train, y_train),
            valid_data=(x_valid, y_valid),
            eta=FixedRate(0.1),
            momentum=FixedMomentum(0),
            num_batches=100,
            max_epochs=100,
            patience=100,
            verbose=True,
            log_data=True,
            )

    # Log the descriptor
    net.log(str(net))

    net.train()


if __name__ == "__main__":
    _main()
