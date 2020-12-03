#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:47:49 2020

@author: tonedogga
"""
import time
import tensorflow as tf
import numpy as np
#a=tf.Variable(42.0)

#tf.print("\n",a.device)


@tf.function
def tf_test(x,y):
    return tf.divide(tf.square(tf.pow(tf.sqrt(x),y)),y)

def tf_test2(x,y):
    return tf.multiply(tf.square(tf.pow(tf.sqrt(x),y)),y)


def np_test(x,y):
    return np.divide(np.square(np.power(np.sqrt(x),y)),y)

def np_test2(x,y):
    return ((np.sqrt(x)**y)**2)/y





start=time.time()
g=tf_test(tf.constant(tf.range(1.)),tf.constant(1.))
tf.print(g)
end=time.time()
tf.print(end-start,"test tf1 seconds")

start=time.time()
g=tf_test(tf.constant(tf.range(100000000.0)),tf.constant(0.98))
#g=tf_test(tf.range(100000000.0),0.98)

tf.print(g)
end=time.time()
tf.print(end-start,"test tf1 seconds")

start=time.time()
g=tf_test2(tf.constant(tf.range(100000000.0)),tf.constant(0.98))
#tf.print(g)
end=time.time()
tf.print(end-start,"test2 tf2 seconds")

start=time.time()
g=tf_test2(tf.constant(tf.range(100000000.0)),tf.constant(0.98))
#tf.print(g)
end=time.time()
tf.print(end-start,"test2 tf2 seconds")

start=time.time()
n=np_test(np.arange(100000000),0.98)
#print(n)
end=time.time()
tf.print(end-start,"1np seconds")

start=time.time()
n=np_test2(np.arange(100000000),0.98)
#print(n)
end=time.time()
tf.print(end-start,"2np seconds")