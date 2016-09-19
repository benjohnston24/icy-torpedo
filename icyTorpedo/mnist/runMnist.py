#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Console script for running mnist"""

# Imports
import sys
import argparse
from icyTorpedo.resources import load_mnist_train_images, \
        load_mnist_train_labels, split_training_data
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.network import baseNetwork
from icyTorpedo.learningrates import FixedRate
from icyTorpedo.linearities import Linear
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Sunday 18 September  22:07:58 AEST 2016'
__license__ = 'MPL v2.0'


def _options(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--description', type=str, dest='desc', default='',
                        help="Description")
    parser.add_argument('-s', '--savename', type=str, dest='name', default='mnist',
                        help="Assign a name to the execution")
    parser.add_argument('-n', '--nodes', type=int, default=700, dest='nodes',
                        help='Number of nodes for each hidden network')
    parser.add_argument('-m', '--max_epochs', type=int, default=1000, dest='epochs',
                        help='Maximum number of epochs')

    args = parser.parse_args()
    return args


def _main(*args, **kwargs):
    args = _options()

    train_images = load_mnist_train_images()
    train_images = train_images.reshape((-1, 28 ** 2))
    train_labels = load_mnist_train_labels()

    x_train, y_train, x_valid, y_valid = split_training_data(train_images,
                                                             train_labels,
                                                             )

    l_input = InputLayer(num_units=28 ** 2, name="Input")
    l_hidden = DenseLayer(input_layer=l_input, num_units=args.nodes, name="Hidden")
    l_output = DenseLayer(input_layer=l_hidden, 
                          num_units=10, 
                          linearity=Linear,
                          name="Output")

    net = baseNetwork(
            network_layers=[l_input, l_hidden, l_output],
            train_data=(x_train, y_train),
            valid_data=(x_valid, y_valid),
            max_epochs=args.epochs,
            name=args.name,
            verbose=True,
            log_data=False,
            )

    # Log the descriptor
    net.log(str(net))

    net.train() 


if __name__ == "__main__":
    _main()
