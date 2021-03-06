#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
from icyTorpedo.layers import iterlayers, addbiasunits
from icyTorpedo.costfunctions import SquaredError
from icyTorpedo.learningrates import FixedRate
from icyTorpedo.momentum import FixedMomentum
import numpy as np
import os
import time
from sklearn.utils import shuffle
from six.moves import cPickle as pickle


__author__ = 'Ben Johnston'
__revision__ = '0.5'
__date__ = 'Wednesday 26 October  09:28:13 AEDT 2016'
__license__ = 'MPL v2.0'


DEFAULT_LOG_EXTENSION = '.log'
DEFAULT_PKL_EXTENSION = '.pkl'
LINE = "-" * 156

MINIBATCHES_OFF = 1

class baseNetwork(object):

    def __init__(self,
                 network_layers=None,
                 train_data=(None, None),
                 valid_data=(None, None),
                 test_data=(None, None),
                 train_data_flipped=(None, None),
                 eta=FixedRate(0.01),
                 momentum=FixedMomentum(0.9),
                 costfunction=SquaredError,
                 num_batches=MINIBATCHES_OFF,  # Set to 1 to di
                 max_epochs=np.inf,
                 patience=100,
                 regression=False,
                 name='neuralNet',
                 verbose=False,
                 log_data=False,
                 *args, **kwargs):

        if network_layers is not None:
            self._setup_layers(network_layers)

        # Define characteristics of the network
        self.cost_function = costfunction()
        self.eta = eta
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.num_batches = num_batches 
        self.patience = patience
        self.verbose = verbose
        self.log_data = log_data
        self.name = name

        # Initialise some variables to be used during training
        self.min_valid_err = np.inf
        self.best_epoch = 0

        # Reserve some variables for storing training data
        self.x_train, self.y_train = train_data
        self.x_valid, self.y_valid = valid_data
        self.x_test, self.y_test = test_data
        self.x_train_flipped, self.y_train_flipped = train_data_flipped

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
        output += "Momentum: %s\n" % str(self.momentum)
        output += "x_train shape: %s\ty_train shape: %s\n" % \
                  (self.x_train.shape, self.y_train.shape)
        output += "x_valid shape: %s\ty_valid shape: %s\n" % \
                  (self.x_valid.shape, self.y_valid.shape)
        output += "Max Epochs: %.0f\n" % self.max_epochs
        output += "Patience: %d\n" % self.patience
        output += "Number minibatches: %d\n" % self.num_batches

        return output

    def forwardprop(self, targets=None, enable_dropout=True):
        """Iterate through each of the layers and compute the activations at each node"""
        for layer in iterlayers(self.output_layer):
            layer.a_h(targets=targets, enable_dropout=enable_dropout)  # Compute the activations

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
        delta_o = delta_o.T
        self.output_layer.delta = delta_o

        # Store for later use
        delta = delta_o

        # Add the bias units of the previous layer
        inputs = addbiasunits(self.output_layer.input_layer.a)

        #self.output_layer.dc_dw = np.dot(inputs.T, delta_o.T)
        self.output_layer.set_dc_dw(np.dot(inputs.T, delta_o.T))

        # Backprop over remaining layers
        for layer_idx in range(2, len(self.network_layers)):

            layer = self.network_layers[-layer_idx]
            layer_after = self.network_layers[-layer_idx + 1]
            layer_before = self.network_layers[-layer_idx - 1]

            # delta = (w^l+1 * delta^(l+1)) * sigma'(h^l)
            # Strip out the biases
            delta = np.dot(delta.T, layer_after.W[1:, :].T) * \
                layer.linearity.prime(layer.h)
            delta = delta.T

            layer.delta = delta

            # Add the bias units to the input
            inputs = addbiasunits(layer_before.a)

            # layer.dc_dw = np.dot(inputs.T, delta.T)
            layer.set_dc_dw(np.dot(inputs.T, delta.T))

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

        # Get the learning rate, if the learning rate changes we only want
        # to do this once per weight update
        learning_rate = self.eta()

        while layer.input_layer is not None:
            update = ((learning_rate * layer.dc_dw) + (self.momentum() * layer.dc_dw_prev))
            layer.dc_dw_prev = update
            layer.W -= update 

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

        self.min_valid_err = np.inf
        self.best_epoch = 0
        epoch = 0

        self.train_err_history = []
        self.valid_err_history = []
        self.correct_class_history = []

        if not self.regression:
            self.log(LINE + "-" * 20)
        else:
            self.log(LINE)

        self.log(self._prepare_train_header())

        if not self.regression:
            self.log(LINE + "-" * 20)
        else:
            self.log(LINE)

        while True:

            # Time the iteration
            start_time = time.time()

            train_err = 0

            # Implement training
            # Shuffle the data
            x_train_shuff = np.copy(self.x_train)
            y_train_shuff = np.copy(self.y_train)

            if (self.x_train_flipped is not None) and (self.y_train_flipped is not None):
                indices_to_flip = np.random.choice(range(len(self.x_train)), 1)

                for idx in indices_to_flip:
                    x_train_shuff[idx] = self.x_train_flipped[idx]
                    y_train_shuff[idx] = self.y_train_flipped[idx]

            x_train_shuff, y_train_shuff = shuffle(x_train_shuff,
                                                   y_train_shuff,
                                                   random_state=int(time.time()))

            for x_batch, y_batch in zip(np.array_split(x_train_shuff, self.num_batches),
                                        np.array_split(y_train_shuff, self.num_batches)):

                # x_train_shuff, y_train_shuff = self.x_train, self.y_train
                self.input_layer.set_inputs(x_batch)
                self.forwardprop(targets=y_batch)

                # Run backprop
                # Reload the training set for updating the weights
                self.backprop(y_batch)
                # TODO: An adaptive update to weights to speed up process
                eta = self.updateweights(y_batch)

            # TODO Redo pseudo - inverse / backprop for whole training set.

            # Predict based on current weights
            train_pred = self.predict(x_train_shuff)

            train_err = self.cost_function(output=train_pred,
                                           target=y_train_shuff)
            self.train_err_history.append(train_err)

            # Check against validation set
            valid_pred = self.predict(self.x_valid)

            valid_err = self.cost_function(output=valid_pred,
                                           target=self.y_valid)
            self.valid_err_history.append(valid_err)

            # If this is a categorisation problem determine if correctly labeled
            if not self.regression:
                correct_class = np.sum(valid_pred.argmax(axis=1) == self.y_valid.argmax(axis=1))
                self.correct_class_history.append(correct_class)

            # End of the iteration

            finish_time = time.time()

            if (valid_err < self.min_valid_err):
                improvement = "*"
                self.min_valid_err = valid_err
                self.best_epoch = epoch
                self.eta.value =  0.95 * self.eta()

                # Cache the best weights
                self.cache_best_weights()

            else:
#                self.eta.value = max(0.0001, 0.995 * self.eta())
                improvement = "{:0.7f}".format(self.min_valid_err)

            iteration_record = \
                "|{:^20}|{:^20}|{:^20}|{:^30}|{:^20}|{:^20}|{:^20}|".format(
                    epoch,
                    "%E" % train_err,
                    "%0.6f" % valid_err,
                    "%E" % (np.cast['float32'](valid_err) / np.cast['float32'](train_err)),
                    "%0.6f" % (finish_time - start_time),
                    improvement,
                    "%0.6E" % eta)

            if not self.regression:
                accuracy_percent = "%0.2f" % (float(correct_class) / float(self.x_valid.shape[0]) * 100)
                iteration_record += "{:^20}|".format(accuracy_percent)

            self.log(iteration_record)

            # Check for early termination
            if (epoch - self.best_epoch) > self.patience:

                self.log("Early Stopping")
                self.log("Best validation error %0.6f @ epoch %d" %
                         (self.min_valid_err, self.best_epoch))
                self.save_network()
                break
            elif epoch > self.max_epochs:
                self.save_network()
                break

            # Increment the iteration counter
            epoch += 1

        if not self.regression:
            return train_err, valid_err, correct_class
        else:
            return train_err, valid_err, None

    def cache_best_weights(self):
        """Cache all the best weights for later storage / use

        Parameters
        -----------

        None

        Returns
        -----------
        None
        """

        # Skip the input layer as no weights present
        for layer in self.network_layers[1:]:
            layer.best_W = np.copy(layer.W)

    def predict(self, inputs):

        self.input_layer.set_inputs(inputs)
        self.forwardprop(enable_dropout=False)

        return self.output_layer.a

    # Save the network
    def save_network(self):
        data_to_save = {
            'weights': [layer.best_W for layer in self.network_layers[1:]]
            # 'network_layers': self.network_layers,
            # 'best_epoch': self.best_epoch,
            # 'min_valid_err': self.min_valid_err,
            # 'train_err_hist': self.train_err_history,
            # 'valid_err_hist': self.valid_err_history,
            # 'correct_class_hist': self.correct_class_history,
        }

        with open(self.save_params_filename, 'wb') as f:
            pickle.dump(data_to_save, f)

    # Load the network
    def load_network(self, filename):

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        for layer, weights in zip(self.network_layers[1:], data['weights']):
            layer.W = weights 

        #self._setup_layers(data['network_layers'])
        if False:
            self.best_epoch = data['best_epoch']
            self.min_valid_err = data['min_valid_err']
            self.train_err_history = data['train_err_hist']
            self.valid_err_history = data['valid_err_hist']
            self.correct_class_history = data['correct_class_hist']

    def _setup_layers(self, network_layers):
        """Setup the layers for use in the object"""

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
