#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
from icyTorpedo.layers import iterlayers
from icyTorpedo.costfunctions import SquaredError
from icyTorpedo.learningrates import FixedRate
from icyTorpedo.linearities import Linear
import numpy as np
import os
import time
from sklearn.utils import shuffle


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Friday 16 September  20:55:50 AEST 2016'
__license__ = 'MPL v2.0'


DEFAULT_LOG_EXTENSION = '.log'
DEFAULT_PKL_EXTENSION = '.pkl'
LINE = "-" * 156


class baseNetwork(object):

    def __init__(self,
                 network_layers=None,
                 train_data=(None, None),
                 valid_data=(None, None),
                 test_data=(None, None),
                 eta=FixedRate(0.01),
                 costfunction=SquaredError,
                 max_epochs=2000,
                 patience=100,
                 regression=False,
                 name='neuralNet',
                 verbose=False,
                 log_data=False,
                 *args, **kwargs):

        # Get the architecture of the network
        self.network_layers = network_layers

        # Define the output layer as the last layer in the list
        self.output_layer = self.network_layers[-1]

        # Determine the input layer by traversing the inputs of the output layer
        layer = self.output_layer

        while layer.input_layer is not None:
            layer = layer.input_layer

        # Define the input layer as the first layer in the list
        self.input_layer = layer 

        # Define characteristics of the network
        self.cost_function = costfunction() 
        self.eta = eta
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.log_data = log_data 
        self.name = name

        # Reserve some variables for storing training data
        self.x_train, self.y_train = train_data 
        self.x_valid, self.y_valid = valid_data 
        self.x_test, self.y_test = test_data 

        # Flag to indicate if the problem is a regression problem
        self.regression = regression

        # Prepare the logfile
        self._prepare_log()

    def __str__(self):

        output = "Neural Network: %s\n" % self.name
        output += "Architecture:\n"

        for layer in self.network_layers:
            output += "%s\n" % str(layer)

        output += "Regression: %s\n" % str(self.regression)
        output += "Cost Function: %s\n" % str(self.cost_function)
        output += "Learning Rate: %s\n" % str(self.eta)
        output += "x_train shape: %s\ty_train shape: %s\n" % \
                (self.x_train.shape, self.y_train.shape)
        output += "x_valid shape: %s\ty_valid shape: %s\n" % \
                (self.x_valid.shape, self.y_valid.shape)
        output += "Max Epochs: %d\n" % self.max_epochs
        output += "Patience: %d\n" % self.patience

        return output

    def forwardprop(self):
        """Iterate through each of the layers and compute the activations at each node"""
        for layer in iterlayers(self.output_layer):
            layer.a_h() # Compute the activations

    def backprop(self, targets):
        """Execute back propogation
        
        First compute the rate of the error at the output 
        delta_o = dC/da * d_linearity 
                = costfunction' * linearity'

        dC_db = delta_o; Change of cost with respect to bias
        dC_dw = np.dot(activation_prev_layer, delta_o)

        The error at other layers can be calculated by

        delta_l = np.dot(weights_prev_layer, delta_prev_layer) * linearity'(h_layer)

        """

        # Start at the output layer
        delta_o = self.cost_function.prime(output=self.output_layer.a,
                                           target=targets) * \
                self.output_layer.linearity.prime(self.output_layer.h)
        self.output_layer.delta = delta_o

        # Store for later use
        delta = delta_o

        # Add the bias units of the previous layer
        input_layer = self.output_layer.input_layer
        inputs = np.hstack((
            np.ones((input_layer.a.shape[0], 1)),
            input_layer.a))

        self.output_layer.dc_dw = np.dot(inputs.T, delta_o) 

        # Backprop over remaining layers
        for layer_idx in range(2, len(self.network_layers)):

            layer = self.network_layers[-layer_idx]
            layer_after = self.network_layers[-layer_idx + 1]
            layer_before = self.network_layers[-layer_idx -1]

            # delta = (w^l+1 * delta^(l+1)) * sigma'(h^l)
            # Strip out the biases
            delta = np.dot(layer_after.W[1:,:], delta.T) * \
                    layer.linearity.prime(layer.h).T

            layer.delta = delta
            # Add the bias units to the input
            inputs = np.hstack((
                np.ones((layer_before.a.shape[0], 1)),
                layer_before.a))

            layer.dc_dw = np.dot(delta, inputs).T 


    def updateweights(self, targets, psi=True):
        """Update the weights of the network in each layer:

        W -= eta * dC/dw
        b -= eta * db/dw

        Parameters
        -----------

        targets:  The target values for the network 
        psi:      Use the pseudo-inverse if possible.  If the output
                  layer has a linear activation and the flag is set
                  to true, use the Moore-Penrose pseudo-inverse to
                  calculate the output layer weights [default: True]

        Returns
        -----------

        The current learning rate used to update the weights

        """

        layer = self.output_layer

        # If the output layer is linear use the pseudo inverse to calculate weights
        # W = TA+ where W are the weights of the output layer, T the target values
        # and A+ the Moore-Penrose inverse of the hidden layer activations
        # P. de Chazal, J. Tapson and A. van Schaik, "A comparison of extreme learning machines 
        # and back-propagation trained feed-forward networks processing the mnist database," 
        # 2015 IEEE International Conference on Acoustics,
        # Speech and Signal Processing (ICASSP), South Brisbane, QLD, 2015, pp. 2165-2168.
        # doi: 10.1109/ICASSP.2015.7178354
        if psi and isinstance(layer.linearity, Linear):
            # Add the bias units to the input
            inputs = np.hstack((
                np.ones((layer.input_layer.a.shape[0], 1)),
                layer.input_layer.a))

            a_plus = np.linalg.pinv(inputs)

            # W = TA+
            layer.W = np.dot(a_plus, targets)

            # Move to next layer
            layer = layer.input_layer


        # Get the learning rate, if the learning rate changes we only want
        # to do this once per weight update
        learning_rate = self.eta()

        while layer.input_layer is not None:
            layer.W -= learning_rate * (layer.dc_dw / targets.shape[0])

            layer = layer.input_layer

        return learning_rate

    def _prepare_train_header(self):
        """Log the required information e.g. column headings prior to training"""

        header = \
             "|{:^20}|{:^20}|{:^20}|{:^30}|{:^20}|{:^20}|{:^20}|".format(
                 "Epoch",
                 "Train Error",
                 "Valid Error",
                 "Valid / Train Error",
                 "Time",
                 "Best Error",
                 "Learning Rate",
                 )

        if not self.regression:
            header += "{:^20}|".format("Accuracy (%)")

        return header


    def train(self):
        """Train the neural network
        
        Simple version, apply each of the training examples in the same order per iteration
        
        """

        min_valid_err = np.inf
        best_epoch = 0

        if not self.regression:
            self.log(LINE + "-" * 20)
        else:
            self.log(LINE)

        self.log(self._prepare_train_header())

        if not self.regression:
            self.log(LINE + "-" * 20)
        else:
            self.log(LINE)

        for epoch in range(self.max_epochs):

            # Time the iteration 
            start_time = time.time()

            train_err = 0

            # Implement training
            # Shuffle the data
            x_train_shuff, y_train_shuff = shuffle(self.x_train,
                                                   self.y_train,
                                                   random_state=int(time.time()))
            #x_train_shuff, y_train_shuff = self.x_train, self.y_train

            # Predict based on current weights
            train_pred = self.predict(x_train_shuff)

            train_err = self.cost_function(output=train_pred,
                                           target=y_train_shuff)

            # Run backprop
            # Reload the training set for updating the weights
            self.backprop(y_train_shuff)
            # TODO: An adaptive update to weights to speed up process
            eta = self.updateweights(y_train_shuff)
 

            # Check against validation set
            valid_pred = self.predict(self.x_valid)

            valid_err = self.cost_function(output=valid_pred,
                                            target=self.y_valid)

               
            # If this is a categorisation problem determine if correctly labeled
            if not self.regression:
                correct_class = np.sum(valid_pred.argmax(axis=1) == self.y_valid.argmax(axis=1))

            # End of the iteration
            finish_time = time.time()

            if (valid_err < min_valid_err):
                improvement = "*"
                min_valid_err = valid_err
                best_epoch = epoch
                self.eta.value *= 1.05  # TODO implement this in a more generic way

            else:
                self.eta.value *= 0.95
                if self.eta.value < 0.000001:
                    self.eta.value = 0.000001
                improvement = ""

            iteration_record = \
                "|{:^20}|{:^20}|{:^20}|{:^30}|{:^20}|{:^20}|{:^20}|".format(
                    epoch,
                    "%0.6f" % train_err,
                    "%0.6f" % valid_err,
                    "%0.6f" % (np.cast['float32'](valid_err) / np.cast['float32'](train_err)),
                    "%0.6f" % (finish_time - start_time),
                    improvement,
                    "%0.6E" % eta)

            if not self.regression:
                accuracy_percent = "%0.2f" % (float(correct_class) / float(self.x_valid.shape[0]) * 100)
                iteration_record += "{:^20}|".format(accuracy_percent)

            self.log(iteration_record)

            # Check for early termination
            if (epoch - best_epoch) > self.patience:

                self.log("Early Stopping")
                self.log("Best validation error %0.6f @ epoch %d" %
                        (min_valid_err, best_epoch))
                break

        if not self.regression:
            return train_err, valid_err, correct_class
        else:
            return train_err, valid_err, None 

    def predict(self, inputs):

        self.input_layer.set_inputs(inputs)
        self.forwardprop()

        return self.output_layer.a

    # Logging functionality
    def _prepare_log(self):
        # Data logging
        # If the log already exists append a .x to the end of the file
        self.log_extension = DEFAULT_LOG_EXTENSION
        self.save_params_extension = DEFAULT_PKL_EXTENSION
        log_basename = self.name
        # log_basename = "%s-%d" % (log_name, hidden_units)
        if os.path.exists("{}{}".format(log_basename, self.log_extension)):
            # See if any other logs exist of the .x format
            log_iter = 0
            while os.path.exists("{}{}.{}".format(log_basename, self.log_extension, log_iter)):
                log_iter += 1

            self.log_filename = "{}{}.{}".format(log_basename, self.log_extension, log_iter)
            self.save_params_filename = "{}{}.{}".format(log_basename, self.save_params_extension, log_iter)
        else:
            self.log_filename = log_basename + self.log_extension
            self.save_params_filename = log_basename + self.save_params_extension

    # Log method
    def log(self, msg):
        if self.verbose:
            print(msg)

        if self.log_data:
            with open(self.log_filename, "a") as f:
                f.write("%s\n" % msg)
