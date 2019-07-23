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
from os.path import join, split, splitext
from configparser import SafeConfigParser as scnf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = scnf()
config.read('config.ini')


class PPREDICT(object):
    '''
    Create gray-scale image for set of color images using trained model.
    '''
    def predict(self, color_img_dir=config.get('directories', 'validate'), img_type=config.get('train', 'img_type')):
        '''
        color_img_dir: Path of folder with color images to test.
        img_type: Type of image, eg: png, jpeg, jpg.
        '''
        try:
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(join('.', config.get(
                    'directories', 'model'), '{}.ckpt-{}.meta'.format(config.get('train', 'model_name'), int(config.get('train', 'epoch_num')) - 1)))
                saver.restore(sess, tf.train.latest_checkpoint(
                    join('.', config.get('directories', 'model'))))

                # Create test dataset
                file_paths = glob.glob(join(
                    '.', color_img_dir, '*.' + img_type))
                test_data = [np.array(cv2.imread(file)) for file in file_paths]
                file_names = list(map(lambda x: splitext(split(x)[-1])[0], file_paths))

                test_dataset = np.asarray(test_data)
                # print(test_dataset.shape)

                # Get graph
                graph = tf.get_default_graph()
                train_op = graph.get_tensor_by_name("aeOutput/BiasAdd:0")
                ae_inputs = graph.get_tensor_by_name("aeInput:0")
                gray_imgs = sess.run(train_op, feed_dict={
                                     ae_inputs: test_dataset})
                for i in range(gray_imgs.shape[0]):
                    cv2.imwrite(join('.', config.get('directories',
                                                     'pred_img_dir'), '{}.{}'.format(file_names[i], img_type)), gray_imgs[i])
        except Exception as e:
            print("Exception: ", e)


if __name__ == "__main__":
    pred_obj = PPREDICT()
    pred_obj.predict()