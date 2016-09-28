#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports

from setuptools import setup, find_packages
from icyTorpedo.__init__ import __version__

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Sunday 18 September  22:49:44 AEST 2016'
__license__ = 'MPL v2.0'

setup(
    name='icyTorpedo',
    description='neural network package',
    url='',
    author='Ben Johnston',
    author_email='',
    version=__version__,
    packages=find_packages(exclude=['tests']),
    entry_points={
        'console_scripts': [
            'basicmnist=icyTorpedo.mnist:_main',
            ],
        },
    # package_data={
    #    '': ['*.pkl'],
    #    },
    # license=open('LICENSE.txt').read(),
    # long_description=open('README.txt').read(),
    )
