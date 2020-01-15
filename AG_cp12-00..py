# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:53:32 2020

@author: Anthony2013
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
# def reset_graph(seed=42):
#     tf.reset_default_graph()
#     tf.set_random_seed(seed)
#     np.random.seed(seed)

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "distributed"


#import tensorflow.compat.v1 as tf
import tensorflow as tf
sess=tf.session()

v1 = tf.Variable(1.0, name="v1")  # pinned to /job:ps/task:0 (defaults to /cpu:0)
v1=v1
c = tf.constant("Hello distributed TensorFlow!")
server = tf.train.Server.create_local_server()
with tf.Session(server.target) as sess:
    print(sess.run(c))


  
    