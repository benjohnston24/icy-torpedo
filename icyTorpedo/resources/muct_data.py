#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Class to load muct data"""

# Imports
import os
import scipy
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import time

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 12 October  10:47:33 AEDT 2016'
__license__ = 'MPL v2.0'

class MUCTData:

    MUCT_IMAGE_FOLDER = os.path.dirname(__file__)
    MUCT_IMAGE_LANDMARKS = os.path.join(MUCT_IMAGE_FOLDER,
                                        'me17-opencv.csv')
    MUCT_FRONTAL_IMAGE_LIST = os.path.join(MUCT_IMAGE_FOLDER,
                                           'muct_frontal_images.txt')
    IMAGE_WIDTH = 480
    IMAGE_HEIGHT = 640

    def set_muct_image_folder(folder):
        if folder is not None:
           MUCTData.MUCT_IMAGE_FOLDER = folder

    def get_muct_image_folder():
        return MUCTData.MUCT_IMAGE_FOLDER

    def yield_frontal_images_list():
        with open(MUCTData.MUCT_FRONTAL_IMAGE_LIST, 'r') as f:
            line = f.readline()
            while line:
                yield line.strip()
                line = f.readline()

    def yield_frontal_images_names():
        with open(MUCTData.MUCT_FRONTAL_IMAGE_LIST, 'r') as f:
            line = f.readline()
            while line:
                yield os.path.splitext(line.strip())[0]
                line = f.readline()

    def read_image(image):
        filename = os.path.join(
                MUCTData.get_muct_image_folder(),
                image)
        img = scipy.misc.imread(
                name=filename,
                flatten=True)
        return img

    def read_images_from_list(images_list=None):
        images = []
        if images_list is None:
            images_list = MUCTData.yield_frontal_images_list()
        for image in images_list: 
            img = MUCTData.read_image(image).flatten()
            images.append(img)
        return np.stack(images, axis=0)

    def read_landmarks_from_list(images_list=None):
        df = pd.read_csv(MUCTData.MUCT_IMAGE_LANDMARKS)
        if images_list is None:
            images_list = MUCTData.yield_frontal_images_names()
        df = df[df['name'].isin(images_list)]
        del df['tag']
        return df

    def test_train_split_names(images_list=None, split_ratio_test=0.1,
                               split_ratio_train=0.7):
        images = []
        if images_list is None:
            images_list = MUCTData.yield_frontal_images_list()

        all_images = [x for x in images_list]

        remaining_images, test_images = train_test_split(
                all_images,
                train_size=(1 - split_ratio_test),
                random_state=int(time.time()))

        train_images, valid_images = train_test_split(
                remaining_images,
                train_size=split_ratio_train,
                random_state=int(time.time()))

        return train_images, valid_images, test_images
