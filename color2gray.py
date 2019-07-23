#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Author: Sarath KS
Date: 21/07/2019
E-mail: sarathks333@gmail.com
'''

import tensorflow as tf
import numpy as np
import cv2
import glob
import os
from configparser import SafeConfigParser as cnf
print("Tensorflow version: {}".format(tf.__version__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = cnf()
config.read('config.ini')

# Creating dependent folders
for w in config.items('directories'):
    if not os.path.exists(w[-1]):
        os.makedirs(w[-1])

COLOR_IMG_DIR = os.path.join('.', config.get('directories', 'source_img_dir'))
TARGET_GRAY_IMG_DIR = os.path.join(
    '.', config.get('directories', 'target_img_dir'))
MODEL_DIR = os.path.join('.', config.get('directories', 'model'))


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


class COLOR2GRAY(object):
    '''
    '''

    def __init__(self, color_img_dir=COLOR_IMG_DIR, target_gray_img_dir=TARGET_GRAY_IMG_DIR, model_dir=MODEL_DIR):
        self.dataset = None
        self.img_type = 'png'
        self.dataset_target = None
        self.color_img_dir = color_img_dir
        self.target_gray_img_dir = target_gray_img_dir
        self.model_dir = model_dir
        self.num_images = 0

    def read_dataset(self, img_type='png'):
        self.img_type = img_type
        img_paths = glob.glob(self.color_img_dir + '*.' + self.img_type)
        self.num_images = len(img_paths)
        self.dataset = np.asarray(
            [np.array(cv2.imread(fname)) for fname in img_paths])
        self.dataset_target = np.asarray([np.array(cv2.imread(fname)) for fname in glob.glob(
            self.target_gray_img_dir + '*.' + self.img_type)])
        # dataset_target = dataset_target[:, :, :, np.newaxis]

    def train(self, batch_size=int(config.get('train', 'batch_size')), epoch_num=int(config.get('train', 'epoch_num'))):
        saving_path = os.path.join(self.model_dir, '{}.ckpt'.format(
            config.get('train', 'model_name')))

        # Loss function
        ae_inputs = tf.placeholder(
            tf.float32, (None, 128, 128, 3), name='aeInput')
        ae_targets = tf.placeholder(
            tf.float32, (None, 128, 128, 3), name='aeTarget')

        # Get Autoencoder tensor
        ae_outputs = Autoencoder(ae_inputs).run()

        # Assign loss function
        loss = tf.reduce_mean(tf.square(ae_outputs - ae_targets))

        # Assingn Adam optimizer
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # Initialize tensor variables
        init = tf.global_variables_initializer()
        # Save models from last two epoches
        saver_ = tf.train.Saver(max_to_keep=2)

        batch_img = self.dataset[0: batch_size]
        batch_out = self.dataset_target[0: batch_size]

        num_batches = self.num_images//batch_size

        sess = tf.Session()
        sess.run(init)

        for ep in range(epoch_num):
            batch_size = 0
            for batch_n in range(num_batches):  # batches loop

                _, c = sess.run([train_op, loss], feed_dict={
                                ae_inputs: batch_img, ae_targets: batch_out})
                print("Epoch: {} - cost = {:.5f}" .format((ep+1), c))

                batch_img = self.dataset[batch_size: batch_size+32]
                batch_out = self.dataset_target[batch_size: batch_size+32]

                batch_size += 32

            saver_.save(sess, saving_path, global_step=ep)

        sess.close()

if __name__ == "__main__":
    obj = COLOR2GRAY()
    obj.read_dataset()
    obj.train()