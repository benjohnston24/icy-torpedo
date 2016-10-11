#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Class to load muct data"""

# Imports
import os
import scipy
import numpy as np
import pandas as pd

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Monday 10 October  22:43:27 AEDT 2016'
__license__ = 'MPL v2.0'

class MUCTData:

    MUCT_IMAGE_FOLDER = os.path.dirname(__file__)
    MUCT_IMAGE_LANDMARKS = os.path.join(MUCT_IMAGE_FOLDER,
                                        'muct76-opencv.csv')
    MUCT_FRONTAL_IMAGE_LIST = os.path.join(MUCT_IMAGE_FOLDER,
                                           'muct_frontal_images.txt')

    def set_muct_image_folder(folder):
        if folder is not None:
           MUCTData.MUCT_IMAGE_FOLDER = folder

    def get_muct_image_folder():
        return MUCTData.MUCT_IMAGE_FOLDER

    def yield_frontal_image_list():
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

    def read_images_from_list():
        images = []
        for image in MUCTData.yield_frontal_image_list():
            img = MUCTData.read_image(image).flatten()
            images.append(img)
        return np.stack(images, axis=0)

    def read_landmarks_from_list():
        df = pd.read_csv(MUCTData.MUCT_IMAGE_LANDMARKS)
        df = df[df['name'].isin(MUCTData.yield_frontal_images_names())]
        del df['tag']
        return df
