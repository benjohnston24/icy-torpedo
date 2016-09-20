#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from unittest.mock import MagicMock, patch, mock_open
from icyTorpedo.network import baseNetwork
from icyTorpedo.layers import InputLayer, DenseLayer
from icyTorpedo.linearities import Sigmoid, Linear
import numpy as np
from io import StringIO


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 September  10:44:02 AEST 2016'
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
                                       linearity=Linear,
                                       name="Output")

        # Set the initial input values
        self.l_in.set_inputs(np.array([[1, 0]]))

        # Construct known weights
        self.l_hidden.W = np.array([
            [0.3, 0.6],
            [0.1, 0.4],
            [0.2, 0.5],])

        self.output_layer.W = np.array([
            [0.9],
            [0.7],
            [0.8]])

        # Target output of network 
        self.target_output = 2.0944 

        self.net = baseNetwork(
                network_layers = [self.l_in, self.l_hidden, self.output_layer],
                targets=self.target_output,
                name='baseNetwork',
                log_data=False,
                verbose=False,
                )

        # Expected output after forward propr 1.904
        self.expected_output = np.array([[
            (s(0.4) * 0.7) + (s(1) * 0.8) + 0.9]])

 
    def test_correct_input_output_layers(self):

        self.assertEqual(self.net.output_layer, self.output_layer)
        self.assertEqual(self.net.input_layer, self.l_in)

    def test_forward_and_back_prop(self):

        self.reset()

        # Execute forward prop
        self.net.forwardprop()

        # Check the outputs are correct
        # Check hidden layer
        np.testing.assert_equal(self.l_hidden.h, np.array([[0.4, 1]]))
        np.testing.assert_allclose(self.l_hidden.a, np.array([[s(0.4), s(1)]]))

        # Check output layer
        np.testing.assert_equal(self.net.output_layer.h, self.expected_output)
        np.testing.assert_equal(self.net.output_layer.a, np.array(s(self.expected_output)))


    def test_backprop(self):

        self.reset()

        self.net.forwardprop()

        # Check back prop
        self.net.backprop(self.target_output)

        # Check output layer
        # delta = (s(1.904) - 2.0944) * s(1.904) * (1 - s(1.904)) 
        # dc_db = delta
        np.testing.assert_approx_equal(self.net.output_layer.delta, np.array([-0.138133]),
                                       significant=4)
        np.testing.assert_approx_equal(self.net.output_layer.dc_db, np.array([-0.138133]),
                                       significant=4)
 
        a_1 = np.array([[s(0.4), s(1)]])
        expected_result = np.dot(a_1.T, -0.138133)
        np.testing.assert_almost_equal(self.net.output_layer.dc_dw, 
                                       expected_result,
                                       decimal=4,
                                       )

        # Check hidden layer
        np.testing.assert_almost_equal(self.l_hidden.delta,
                                       np.array([[-0.02323156, -0.02172688]]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.dc_db,
                                       np.array([[-0.02323156, -0.02172688]]),
                                       decimal=4,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.dc_dw,
                                       np.array([[-0.02323156, -0.02172688], [0,0]]),
                                       decimal=4,
                                       )

    def test_update_weights(self):

        self.reset()

        self.net.forwardprop()
        self.net.backprop(self.target_output)

        self.net.updateweights()

        # Check weights

        np.testing.assert_almost_equal(self.net.output_layer.W,
                                       np.array([
                                           [0.7000827],
                                           [0.8010098]]),
                                       decimal=3,
                                       )
        np.testing.assert_almost_equal(self.net.output_layer.b,
                                       np.array([
                                           [0.90138133]]),
                                       decimal=3,
                                       )

        np.testing.assert_almost_equal(self.l_hidden.W,
                                       np.array([
                                           [0.100232316, 0.4],
                                           [0.200217269, 0.5]]),
                                       decimal=3,
                                       )
        np.testing.assert_almost_equal(self.l_hidden.b,
                                       np.array([
                                           [0.300232316, 0.600217269]]),
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

    @patch('os.path.exists', log_file_mock_two_files)
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


