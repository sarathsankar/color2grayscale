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
from configparser import SafeConfigParser
from autoencoder import Autoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = SafeConfigParser()
config.read('config.ini')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join('.', config.get('directories', 'model'), 'ColorToGray.ckpt-{}.meta'.format( int(config.get('train', 'epoch_num')) - 1 )))
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join('.', config.get('directories', 'model'))))
    # ae_inputs = sess.run('input')
    # print(ae_inputs)

    print("Done...")

    # For reading the file contents. 

    filenames = glob.glob('./' + config.get('directories', 'color_img_dir') + '*.png')

    test_data = []
    for file in filenames[0:100]:
        test_data.append(np.array(cv2.imread(file)))

    test_dataset = np.asarray(test_data)
    print(test_dataset.shape)

    # Running the test data on the autoencoder
    batch_imgs = test_dataset
    
    graph = tf.get_default_graph()
    # ae_inputs = graph.get_tensor_by_name("aeInput:0")
    xx = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for i in xx:
        print(i)
    train_op = graph.get_tensor_by_name("aeOutput/BiasAdd:0")
    ae_inputs = graph.get_tensor_by_name("aeInput:0")
    # ae_targets = graph.get_tensor_by_name("aeTarget:0")
    # y_pred_img = np.zeros((None, 128, 128, 3))
    gray_imgs = sess.run(train_op, feed_dict = {ae_inputs: batch_imgs})
    print(">>>>>>>>>>>>")
    # sess.run(init)
    for i in range(gray_imgs.shape[0]):
        cv2.imwrite('./' + config.get('directories', 'gray_pred_img_dir') + str(i) + '.png', gray_imgs[i])
