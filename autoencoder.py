#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Author: Sarath KS
Date: 21/07/2019
E-mail: sarathks333@gmail.com
'''

import tensorflow as tf


class Autoencoder(object):
    '''
    '''
    def __init__(self, input):
        self.input = input

    def run(self):
        # Encoder
        nnet = tf.layers.conv2d(self.input, 128, 2, activation=tf.nn.relu)
        nnet = tf.layers.max_pooling2d(nnet, 2, 2, padding='same')

        # Decoder
        nnet = tf.image.resize_nearest_neighbor(nnet, [129, 129])
        nnet = tf.layers.conv2d(nnet, 1, 2, activation=None, name='aeOutput')
        return nnet
