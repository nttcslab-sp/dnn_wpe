#!/usr/bin/env python
from setuptools import setup

setup(name='pytorch_wpe',
      version='0.0.0',
      description='A pytorch implementation of Weighted Prediction Error',
      author='',
      author_email='',
      url='',
      packages=['pytorch_wpe'],
      install_requires=['numpy', 'torch'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest-cov', 'pytest-html', 'pytest']
      )
