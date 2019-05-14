#!/usr/bin/env python
# coding=utf-8
""" An example showcasing the logging system of Sacred."""
from __future__ import division, print_function, unicode_literals
import logging
from sacred import Experiment

ex = Experiment('log_example')

# set up a custom logger
logger = logging.getLogger('mylogger')
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')


@ex.config
def cfg():
    number = 2
    got_gizmo = False


@ex.automain
def main(number):
    logger.info('aaa')
