#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import os
import pandas
from sklearn.cross_validation import train_test_split
import time
import numpy as np
import gzip
import pandas as pd
from random import shuffle
from six.moves import cPickle as pickle

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '19-Aug-2016 16:00:09 AEST'
__license__ = 'MPL v2.0'


RESOURCE_DIR = os.path.dirname(__file__)
## MNIST
MNIST_FOLDER = '/home/ben/datasets/MNIST'
DEFAULT_TRAIN_SET = os.path.join(MNIST_FOLDER, 'training.csv')
MNIST_TRAIN_IMAGES = os.path.join(MNIST_FOLDER, 'train-images-idx3-ubyte.gz')
MNIST_TRAIN_LABELS = os.path.join(MNIST_FOLDER, 'train-labels-idx1-ubyte.gz')
MNIST_TEST_IMAGES = os.path.join(MNIST_FOLDER, 't10k-images-idx3-ubyte.gz')
MNIST_TEST_LABELS = os.path.join(MNIST_FOLDER, 't10k-labels-idx1-ubyte.gz')
MNIST_IMAGE_SIZE = 28 ** 2
MNIST_NUMBER_LABELS = 10

#CIFAR10
CIFAR10_FOLDER = '/home/ben/datasets/CIFAR10'
CIFAR10_TRAIN_DATA = [os.path.join(CIFAR10_FOLDER, 'data_batch_%d.bin' % i) for i in range(1,6)]
CIFAR10_TEST_DATA = os.path.join(CIFAR10_FOLDER, 'test_batch.bin')

SPECIAL_LANDMARKS_NPY = "_specialised_landmarks.pkl"

KAGGLE_LANDMARKS = (
        ('left_eye_center_x', 'left_eye_center_y'),
        ('right_eye_center_x', 'right_eye_center_y'),
        ('left_eye_inner_corner_x', 'left_eye_inner_corner_y'),
        ('left_eye_outer_corner_x', 'left_eye_outer_corner_y'),
        ('right_eye_inner_corner_x', 'right_eye_inner_corner_y'),
        ('right_eye_outer_corner_x', 'right_eye_outer_corner_y'),
        ('left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y'),
        ('left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y'),
        ('right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y'),
        ('right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y'),
        ('nose_tip_x', 'nose_tip_y'),
        ('mouth_left_corner_x', 'mouth_left_corner_y'),
        ('mouth_right_corner_x', 'mouth_right_corner_y'),
        ('mouth_center_top_lip_x', 'mouth_center_top_lip_y'),
        ('mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'),
        )

def generate_specialised_datasets():

    # Load the data frame
    df = pd.read_csv(DEFAULT_TRAIN_SET)
    df['idx'] = pd.Series(range(len(df.Image)), index=df.index, dtype=np.int)

    # Get the total number of valid landmarks of each type
    number_valid_landmarks = []
    for idx, cols in enumerate(KAGGLE_LANDMARKS):
        d_col = df[cols[0]].dropna()
        number_valid_landmarks.append(len(d_col))

    # Create the common test set
    train_test_ratio = 1  # 0.97
    number_test_samples = int((1 - train_test_ratio) * len(df))

    # Ensure all landmarks are present
    df_test = df.dropna()

    # Get the ids and shuffle
    df_test_idx = df_test.idx.values
    shuffle(df_test_idx)

    test_idx = df_test_idx[:number_test_samples]
    print("-" * 90)
    print("Number of specialist samples")
    print("Number of test samples: %d" % len(test_idx))
    print("-" * 90)
    print("{:<40}{:^10}{:^10}{:^10}{:^10}{:^10}".format(
        "Feature", "Train", "Valid", "Test", "Sum", "#Avail"))
    print("-" * 90)

    # Remove the test data from the data frame
    df = df[~df.idx.isin(test_idx)]

    for idx, cols in enumerate(KAGGLE_LANDMARKS):
        cols_with_idx = [col for col in cols]
        cols_with_idx.append('idx')

        data = df.loc[:,cols_with_idx]
        data = data.dropna()

        train_idx, valid_idx = train_test_split(data.idx.values, train_size=0.7)
        print("{:<40}{:^10}{:^10}{:^10}{:^10}{:^10}".format(
            cols[0], len(train_idx), len(valid_idx),
            len(test_idx), len(train_idx) + len(valid_idx) + len(test_idx), 
            number_valid_landmarks[idx],
            ))

        filename = "{}{}".format(idx, SPECIAL_LANDMARKS_NPY)
        filename = os.path.join(RESOURCE_DIR, filename)

        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    'idx' : idx,
                    'train_idx': train_idx,
                    'valid_idx': valid_idx,
                    'test_idx': test_idx,
                },f)

def load_prepared_indices(landmark=0):
    """Load the pre split specialised training data indices

    parameters
    -----------

    landmark :  The landmark number for the specialised data set

    returns
    -----------
    A tuple of numpy arrays containing the (training_indices, validation_indices, test_indices) 
    """
    filename = os.path.join(RESOURCE_DIR,
                            "%d%s" % ( landmark, SPECIAL_LANDMARKS_NPY))

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if int(data['idx']) != landmark:
        raise ValueError("Landmark {} data was loaded, not landmark {}".format(
            int(data['idx']),
            landmark,
            ))
    return (list(data['train_idx']), 
            list(data['valid_idx']), 
            list(data['test_idx']))

def load_from_kaggle_by_index(index=0, cols=None):
    """Load image and landmark information from the DEFAULT_TRAIN_SET

    parameters
    -----------

    index :  The index of the data within DEFAULT_TRAIN_SET

    returns
    -----------
    A tuple of numpy arrays where the first element is the numpy image vector and the second the landmark
    """
    df = pd.read_csv(DEFAULT_TRAIN_SET)

    if isinstance(index, list):
        x = [np.fromstring(im, sep=' ') for im in df.loc[index,'Image']]
        x = np.vstack(x)
    else:
        x = np.fromstring(df.loc[index, 'Image'], sep=' ')
        x = x.reshape((-1, x.shape[0]))

    if cols is None:
        cols = df.columns.tolist()
        cols.pop(cols.index('Image'))

    y = df.loc[index, cols].values

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    if np.any(np.isnan(y)):
        raise ValueError("At least one nan present in landmarks data")

    if not isinstance(index, list):
        y = y.reshape((-1, y.shape[0]))

    # Centre the data about the mean
    x, x_max, x_mean = _scale_data(x)
    y, y_max, y_mean = _scale_data(y)

    return x, y, (x_max, x_mean, y_max, y_mean)

def load_training_data(filename=DEFAULT_TRAIN_SET):
    """Load the training set

    Parameters
    ------------

    filename : the filename of the data set to load

    Returns
    ------------

    a pandas DataFrame containing the data
    """

    data = pandas.read_csv(filename)
    return data

def remove_incomplete_data(data):
    return data.dropna()

def _scale_data(data_in):

    x = data_in
    data_max = np.max(data_in)
    x = x / data_max

    data_mean = np.mean(x)
    x = x - data_mean

    return x, data_max, data_mean
    
def extract_image_landmarks(data_in):

    # Convert the images
    data_in['Image'] = \
            data_in['Image'].apply(
            lambda im: np.fromstring(im, sep=' '))

    # Extract the images
    # Scale the images
    x = np.vstack(data_in['Image'].values)
    x, x_max, x_mean = _scale_data(x)
    x = x.astype(np.float32)

    # Extract the labels
    labels = data_in.columns.tolist()
    labels.pop(labels.index('Image'))
    y = data_in[labels].values
    y, y_max, y_mean = _scale_data(y)
    y = y.astype(np.float32)

    return x, y, (x_max, x_mean, y_max, y_mean)


def split_training_data(x, y, split_ratio=0.7):

    x_train, x_valid, y_train, y_valid = \
        train_test_split(x, y,
                         train_size=split_ratio,
                         random_state=int(time.time()))

    return x_train, y_train, x_valid, y_valid


def load_data(filename=DEFAULT_TRAIN_SET, dropna=True, split_ratio=0.7):
    """Load the training set, extract features and split into training and validation groups

    Parameters
    ------------

    filename : (optional) the filename of the data set to load, default set to DEFAULT_TRAIN_SET
    dropna   : (optional) remove samples with incomplete data from the original data set.  Set to True by default, det
               to False to keep all data
    split_ratio : (optional) the train / validation ratio for the data set.  0.7 by default (70% of the data is used for
                  the training set)

    Returns
    ------------

    a list of np.arrays containing the training and validation sets 
    [train_in, train_targets, valid_in, valid_targets]
    and the maximum and mean values of the input and target training data
    (in_max, in_mean, target_max, target_mean)
    """
    data = load_training_data(filename)
    if dropna:
        data = remove_incomplete_data(data)
    x, y, offsets = extract_image_landmarks(data)
    return split_training_data(x, y, split_ratio=split_ratio), offsets


def _read(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def load_mnist_train_images():
    """Load the MNIST training set images

    Parameters
    ------------

    None

    Returns
    ------------
    The MNIST training set images as a numpy array with shape (number of images, rows, cols)

    """
    return load_mnist_images(MNIST_TRAIN_IMAGES)


def load_mnist_train_labels():
    """Load the MNIST training set labels

    Parameters
    ------------

    None

    Returns
    ------------
    The MNIST training set labels using one hot encoding  as a numpy array with shape (number of images, 10)

    """
    return load_mnist_labels(MNIST_TRAIN_LABELS)


def load_mnist_test_images():
    """Load the MNIST test set images

    Parameters
    ------------

    None

    Returns
    ------------
    The MNIST test set images as a numpy array with shape (number of images, rows, cols)

    """
    return load_mnist_images(MNIST_TEST_IMAGES)


def load_mnist_test_labels():
    """Load the MNIST test set labels

    Parameters
    ------------

    None

    Returns
    ------------
    The MNIST test set labels using one hot encoding  as a numpy array with shape (number of images, 10)

    """
    return load_mnist_labels(MNIST_TEST_LABELS)


def load_mnist_images(filename=MNIST_TRAIN_IMAGES):
    """Load a set of MNIST images
    This function extracts the MNIST images contained within a gzip file.


    Parameters
    ------------

    filename : (optional) the filename of the data set to load, default set to nnet.resources.MNIST_TRAIN_IMAGES

    Returns
    ------------
    The images as a numpy array with shape (number of images, rows, cols)

    """

    with gzip.open(filename, 'rb') as bytestream:
        magic = _read(bytestream)

        # Check correct magic number
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in %s' %
                (magic, filename))

        num_images = _read(bytestream)
        rows = _read(bytestream)
        cols = _read(bytestream)
        buf = bytestream.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        data = data.astype(np.float32)

        # Scale the data
        data /= 255

        # Shift to be between -1 and 1
        # Subtract the mean data
        mu = np.mean(data)
        data -= mu
        return data


def load_mnist_labels(filename=MNIST_TRAIN_LABELS):
    """Load a set of MNIST labels
    This function extracts the MNIST labels contained within a gzip file.

    Parameters
    ------------

    filename : (optional) the filename of the data set to load, default set to nnet.resources.MNIST_TRAIN_LABELS

    Returns
    ------------
    The labels using one hot encoding as a numpy array with shape (number of samples, 10)

    """
    with gzip.open(filename, 'rb') as bytestream:
        magic = _read(bytestream)

        # Check correct magic number
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in %s' %
                (magic, filename))

        num_labels = _read(bytestream)
        buf = bytestream.read(num_labels)
        data = np.frombuffer(buf, dtype=np.uint8)
        # Return with one hot encoding
        encoding = np.zeros((num_labels, MNIST_NUMBER_LABELS))
        for idx, label in enumerate(data):
            encoding[idx, label] = 1
        encoding = encoding.astype(np.float32)
        return encoding

def _read_cifar10(filename):

    SAMPLES_PER_FILE = 1000

    labels = []
    images = []

    with open(filename, "rb") as f:
        for i in range(SAMPLES_PER_FILE):
            labels.append(np.frombuffer(f.read(1), np.uint8)[0])
            channels = []
            for j in range(len("RGB")):
                channels.append(np.frombuffer(f.read(1024), np.uint8))
                channels[-1] = channels[-1].reshape((32,32))
            images.append(np.stack(channels, axis=2))

    return labels, images

def load_cifar10_data(data=CIFAR10_TRAIN_DATA):

    train_labels = []
    train_images = []

    # According to http://www.cs.toronto.edu/~kriz/cifar.html 
    # each training file contains 1000 images and labels 
    for filename in data:
        labels, images = _read_cifar10(filename)
        train_labels.extend(labels)
        train_images.extend(images)

    return np.array(train_labels).reshape((-1, 1)), np.array(train_images)
        
def load_cifar10_train_data(data=CIFAR10_TRAIN_DATA):
    return load_cifar10_data(data)

def load_cifar10_test_data(data=[CIFAR10_TEST_DATA]):
    return load_cifar10_data(data)



if __name__ == "__main__":
    generate_specialised_datasets()
