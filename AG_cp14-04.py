# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:30:03 2020

@author: Anthony Paech 2016
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
 
try:
  import tensorflow.compat.v1 as tf
except Exception:
  pass

# #tf.enable_eager_execution()
# tf.compat.v1.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )

# tf.enable_v2_behavior()



#import tensorflow as tf

#import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("tensorflow:",tf.__version__)
   



#reset_graph()


reset_graph()


n_steps = 3
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)

seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                    sequence_length=seq_length)
    
    
init = tf.global_variables_initializer()

X_batch = np.array([
        # step 0     step 1
        [[0, 1, 2], [9, 8, 7],[9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0],[0,0,0]], # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4],[0,0,0]], # instance 3
        [[9, 0, 1], [3, 2, 1],[0,0,0]] # instance 4
    ])
seq_length_batch = np.array([3, 1, 2, 2])



with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
    
    
    
print(outputs_val)
print(states_val)