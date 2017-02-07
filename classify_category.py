#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#import sys
#sys.path.insert(1,'/usr/local/lib/python2.7/dist-packages')

import tensorflow as tf
import numpy as np

import os
import glob

data_basedir = "/media/Videos/youtube8m"
video_data_dir = os.path.join(data_basedir, "yt8m_video_level")

for shard in glob.glob(os.path.join(video_data_dir, "*.tfrecord")):
    for serialized_example in tf.python_io.tf_record_iterator(shard):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        print "[Debug] video_id={}, labels={}".format(
            example.features.feature["video_id"].bytes_list.value,
            example.features.feature["labels"].int64_list.value,
        )