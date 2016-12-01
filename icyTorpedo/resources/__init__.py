#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

from .data import (
        RESOURCE_DIR,
        RESOURCE_DIR as DATA_FOLDER,
        DEFAULT_TRAIN_SET,
        MNIST_TRAIN_IMAGES,
        MNIST_TRAIN_LABELS,
        MNIST_TEST_IMAGES,
        MNIST_TEST_LABELS,
        MNIST_NUMBER_LABELS,
        MNIST_IMAGE_SIZE,
        KAGGLE_LANDMARKS,
        load_training_data,
        remove_incomplete_data,
        extract_image_landmarks,
        split_training_data,
        load_data,
        load_mnist_train_images,
        load_mnist_train_labels,
        load_mnist_test_images,
        load_mnist_test_labels,
        generate_specialised_datasets,
        load_prepared_indices,
        load_from_kaggle_by_index,
        load_cifar10_train_data,
        load_cifar10_test_data,
        load_cifar10_meta_data,
        )

from .muct_data import (
        MUCTData,
        )

from .bioid_data import (
        BioIdData,
        )
