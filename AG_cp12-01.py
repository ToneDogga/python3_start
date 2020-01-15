# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:16:51 2020

@author: Anthony2013
"""
from __future__ import absolute_import, division, print_function, unicode_literal

try:
  import tensorflow.compat.v2 as tf
except Exception:
  pass

tf.enable_v2_behavior()

print(tf.__version__)

#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def variables_on_cpu(op):
    if op.type=="Variable":
        return "/cpu:0"
    else:
        return "/gpu:0"
    

#initializer = tf.initializers.random_normal(-1, 1)

    
with tf.device("/gpu:0"):   #variables_on_cpu):       
    rand_t = tf.random_uniform([5], 0, 10, dtype=tf.int32, seed=42)
    a = tf.Variable(rand_t,name="a")
    b = tf.Variable(rand_t,name="b")
    c=tf.add(a,b)
#    init=tf.global_variables_initialiser()
    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("two",c.eval())
