#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Class to load bioid data"""

# Imports
import os
import scipy
import numpy as np
import pandas as pd
import sklearn
if sklearn.__version__ != "0.17.1":
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split
import time


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Friday 14 October  06:53:20 AEDT 2016'
__license__ = 'MPL v2.0'


class BioIdData:

    IMAGE_FOLDER = '/home/ben/Workspace/bioid/faces' 
    #IMAGE_FOLDER = '/home/ben/Workspace/bioid/images' 
    LANDMARKS_FOLDER = '/home/ben/Workspace/bioid/faces_landmarks' 
    #LANDMARKS_FOLDER = '/home/ben/Workspace/bioid/points_20' 

    def load_names(image_folder=IMAGE_FOLDER):

        names = []
        for filename in os.listdir(image_folder):

            stem, ext = os.path.splitext(filename)

            if stem not in names:
                names.append(stem)

        return names

    def load_images(names_list, extension='.pgm'):

        images = []
        for filename in names_list: 

            img = scipy.misc.imread(
                    name=os.path.join(BioIdData.IMAGE_FOLDER,
                                      '%s%s' % (filename, extension)),
                    flatten=True)
            images.append(img)

        images = np.stack(images, axis=0)

        return images

    def read_landmark_file(filename):

        with open(filename, 'r') as f:
            data = f.read()

        data = data.split('\n')

        points = []
        for coords in data[3:-2]:
            points.append([float(x) for x in coords.split(' ')])

        return points


    def load_landmarks(names_list):

        landmarks = []

        for name in names_list:

            points = BioIdData.read_landmark_file(
                    os.path.join(BioIdData.LANDMARKS_FOLDER,
                                 '%s.pts' % name.lower()))
            landmarks.append(points)

        landmarks = np.stack(landmarks, axis=0)

        return landmarks

    def scale_data(x):

        max_x = np.max(x)

        x /= max_x

        mean_x = np.mean(x)

        x -= mean_x

        return x, max_x, mean_x
