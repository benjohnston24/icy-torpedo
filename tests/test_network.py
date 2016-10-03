#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Test the network module"""

# Imports
import unittest
from unittest.mock import MagicMock, patch, mock_open
from icyTorpedo.network import baseNetwork
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.linearities import Sigmoid
import numpy as np
from io import StringIO


__author__ = 'Ben Johnston'
__revision__ = '0.2'
__date__ = 'Monday 3 October  21:18:40 AEDT 2016'
__license__ = 'MPL v2.0'


s = Sigmoid()
log_file_mock_one_file = MagicMock(side_effect=[True, False])
log_file_mock_two_files = MagicMock(side_effect=[True, True, False])
mock_pickle = MagicMock()
mock_file_open = mock_open()


class TestNetwork(unittest.TestCase):

    def setUp(self):

        self.reset()

    def reset(self):
        # Define the architecture of the network
        self.l_in = InputLayer(num_units=2, name="Input")
        self.l_hidden = DenseLayer(input_layer=self.l_in, num_units=2, name="Hidden")
        self.output_layer = DenseLayer(input_layer=self.l_hidden,
                                       num_units=1,
                                       name="Output")

        # Set the initial input values
        self.l_in.set_inputs(np.array([[0, 1]]))

        # Construct known weights
        self.l_hidden.W = np.array([
            [0.5, 0.5],
            [0.1, 0.2],
            [0.3, 0.4],
            ])

        self.output_layer.W = np.array([
            [0.03],
            [0.01],
            [0.02]])

        # Target output of network
        self.target_output = np.array([[1]])

        self.net = baseNetwork(
                network_layers=[self.l_in, self.l_hidden, self.output_layer],
                targets=self.target_output,
                name='baseNetwork',
                log_data=False,
                verbose=False,
                )

        # Expected output after forward propr 1.02
        self.expected_output_h = np.array([[0.051118]])
        self.expected_output_a = np.array([[0.51278]])

    def test_correct_input_output_layers(self):

        self.assertEqual(self.net.output_layer, self.output_layer)
        self.assertEqual(self.net.input_layer, self.l_in)

    def test_forwardprop(self):

        self.reset()

        # Execute forward prop
        self.net.forwardprop()

        # Check the outputs are correct
        # Check hidden layer
        np.testing.assert_equal(self.l_hidden.h, np.array([[0.8, 0.9]]))
        np.testing.assert_allclose(self.l_hidden.a, np.array([[s(0.8), s(0.9)]]))

        # Check output layer
        np.testing.assert_almost_equal(
                self.net.output_layer.h,
                self.expected_output_h,
                decimal=3,
                )
        np.testing.assert_almost_equal(
                self.net.output_layer.a,
                self.expected_output_a,
                decimal=3,
                )

    def test_backprop(self):

        self.reset()

        self.net.forwardprop()

        # Check back prop
        self.net.backprop(self.target_output)

        # Check output layer
        # delta_o = (a_o - t) * a_o * (1 - a_o)
        np.testing.assert_approx_equal(self.net.output_layer.delta, np.array([-0.1217]),
                                       significant=2)
        np.testing.assert_almost_equal(self.net.output_layer.dc_dw,
                                       np.array([
                                           [-0.1217],
                                           [-0.08397],
                                           [-0.08652]
                                       ]),
                                       decimal=4,
                                       )

        # Check hidden layer
        np.testing.assert_almost_equal(self.l_hidden.delta,
                                       np.array([[-0.00026038],
                                                 [-0.0005003]]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.dc_dw,
                                       np.array([
                                           [-0.00026038, -0.0005003],
                                           [0, 0],
                                           [-0.00026038, -0.0005003],
                                           ]),
                                       decimal=4,
                                       )

    def test_update_weights(self):

        self.reset()

        self.net.forwardprop()
        self.net.backprop(self.target_output)

        self.net.updateweights(self.target_output)

        # Check weights

        np.testing.assert_almost_equal(self.net.output_layer.W,
                                       np.array([
                                           [0.031217],
                                           [0.0108397],
                                           [0.0208652],
                                           ]),
                                       decimal=3,
                                       )

        np.testing.assert_almost_equal(self.l_hidden.W,
                                       np.array([
                                           [0.5, 0.5],
                                           [0.1, 0.2],
                                           [0.3, 0.4],
                                           ]),
                                       decimal=3,
                                       )

    def test_log_file_name(self):
        """Check the log filenames are correctly established"""
        self.assertEqual(self.net.log_filename, 'baseNetwork.log')
        self.assertEqual(self.net.save_params_filename, 'baseNetwork.pkl')

    @patch('os.path.exists', log_file_mock_one_file)
    def test_prepare_log_exists(self):
        """Check the log filenames are correctly established - a log of the same name already exists"""

        self.reset()

        self.assertEqual(self.net.log_filename, "baseNetwork.log.0",
                         "incorrect log filename")
        self.assertEqual(self.net.save_params_filename, "baseNetwork.pkl.0",
                         "incorrect pickle filename")

    @patch('os.path.exists', log_file_mock_two_files)  # noqa: F811
    def test_prepare_log_exists(self):
        """Check the log filenames are correctly established - two logs of the same name already exist"""

        self.reset()
        self.assertEqual(self.net.log_filename, "baseNetwork.log.1",
                         "incorrect log filename")
        self.assertEqual(self.net.save_params_filename, "baseNetwork.pkl.1",
                         "incorrect pickle filename")

    def test_log_corrects_data(self):
        """Check the correct data is stored in the log"""

        self.net.log_data = True
        mock_log = mock_open()

        with patch('builtins.open', mock_log):
            self.net.log('test some info')

        mock_log.assert_called_with(self.net.log_filename, 'a')
        handle = mock_log()
        handle.write.assert_called_once_with('test some info\n')

    def test_str(self):

        self.reset()

        self.net.network_layers = ["Input", "Hidden", "Output"]
        self.net.cost_function = "SquaredError"
        self.net.eta = 'FixedRate: 1.000000E-03'

        self.net.x_train = np.zeros((128, 28 ** 2))
        self.net.y_train = np.zeros((128, 10))

        self.net.x_valid = np.zeros((78, 28 ** 2))
        self.net.y_valid = np.zeros((78, 10))

        expected_output = "Neural Network: baseNetwork\n"\
                          "Architecture:\nInput\nHidden\nOutput\n"\
                          "Regression: False\n"\
                          "Cost Function: SquaredError\n"\
                          "Learning Rate: FixedRate: 1.000000E-03\n"\
                          "x_train shape: (128, 784)\ty_train shape: (128, 10)\n"\
                          "x_valid shape: (78, 784)\ty_valid shape: (78, 10)\n"\
                          "Max Epochs: 2000\n"\
                          "Patience: 100\n"

        self.assertEqual(str(self.net), expected_output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_verbose(self, mock_stdout):

        self.reset()

        self.net.verbose = True

        self.net.log("mock stdout")

        self.assertEqual("mock stdout\n", mock_stdout.getvalue())

    def test_prepare_train_header(self):

        header = self.net._prepare_train_header()

        expected_result = \
            "|{:^20}|{:^20}|{:^20}|{:^30}|{:^20}|{:^20}|{:^20}|{:^20}|".format(
                 "Epoch",
                 "Train Error",
                 "Valid Error",
                 "Valid / Train Error",
                 "Time",
                 "Best Error",
                 "Learning Rate",
                 "Accuracy (%)",
                 )

        self.assertEqual(header, expected_result)


class TestMultipleSamples(unittest.TestCase):

    def setUp(self):

        self.reset()

    def reset(self):
        # Define the architecture of the network
        self.l_in = InputLayer(num_units=2, name="Input")
        self.l_hidden = DenseLayer(input_layer=self.l_in, num_units=2, name="Hidden")
        self.output_layer = DenseLayer(input_layer=self.l_hidden,
                                       num_units=1,
                                       name="Output")

        # Set the initial input values
        x_test = np.array([
            [1, 0],
            [0, 1],
            ])

        self.l_in.set_inputs(np.array(x_test))

        # Construct known weights
        self.l_hidden.W = np.array([
            [0.5, 0.5],
            [0.1, 0.2],
            [0.3, 0.4],
            ])

        self.output_layer.W = np.array([
            [0.03],
            [0.01],
            [0.02]])

        # Target output of network
        self.target_output = np.array([
            [1],
            [1],
            ])

        self.net = baseNetwork(
                network_layers=[self.l_in, self.l_hidden, self.output_layer],
                targets=self.target_output,
                name='baseNetwork',
                log_data=False,
                verbose=False,
                )

    def test_forwardprop(self):

        self.reset()

        self.net.forwardprop()

        np.testing.assert_almost_equal(
                self.l_hidden.h,
                np.array([
                    [0.6, 0.7],
                    [0.8, 0.9],
                    ]),
                decimal=2)

        np.testing.assert_almost_equal(
                self.net.output_layer.h,
                np.array([
                    [0.0498],
                    [0.05432],
                    ]),
                decimal=2)

    def test_backprop(self):

        self.reset()

        self.net.forwardprop()

        # Check back prop
        self.net.backprop(self.target_output)

        # Check output layer
        # delta_o = (a_o - t) * a_o * (1 - a_o)
        np.testing.assert_almost_equal(self.net.output_layer.delta,
                                       np.array([
                                           [-0.1218126],
                                           [-0.12176264],
                                           ]),
                                       decimal=2)

        np.testing.assert_almost_equal(self.net.output_layer.dc_dw,
                                       np.array([
                                           [-0.24354],
                                           [-0.16263621],
                                           [-0.16793401]
                                       ]),
                                       decimal=4,
                                       )

        # Check hidden layer
        np.testing.assert_almost_equal(self.l_hidden.delta,
                                       np.array([
                                           [-0.00027868, -0.00026038],
                                           [-0.00054014, -0.0005003],
                                           ]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.dc_dw,
                                       np.array([
                                           [0, 0],
                                           [0, 0],
                                           [0, 0],
                                           ]),
                                       decimal=1,
                                       )

    def test_update_weights(self):

        self.reset()

        self.net.forwardprop()
        self.net.backprop(self.target_output)

        self.net.updateweights(self.target_output)

        # Check weights

        np.testing.assert_almost_equal(self.net.output_layer.W,
                                       np.array([
                                           [0.03],
                                           [0.01],
                                           [0.02],
                                           ]),
                                       decimal=2,
                                       )

        np.testing.assert_almost_equal(self.l_hidden.W,
                                       np.array([
                                           [0.5, 0.5],
                                           [0.1, 0.2],
                                           [0.3, 0.4],
                                           ]),
                                       decimal=1,
                                       )
