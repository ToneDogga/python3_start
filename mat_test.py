# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np

# def runningMeanFast(x, N):
# #    return tf.convolve(x, np.ones((N,))/N)[(N-1):]
#     return tf.nn.convolution(x, np.ones((N,))/N)[(N-1):]



# def mat_add(series,mat_days):
#     result=tf.Variable(tf.zeros((series.shape[0],series.shape[1]),dtype=tf.int32))
#  #   rm=np.rolling_apply(series, lambda x: np.mean(x), window=10)
#    # squares = tf.map_fn(lambda x: x * x, series)
#     return squares



#ones_1d = np.ones(4)
def mat_add_1d(series,mat_days):
    weight_1d = np.ones(mat_days)
    strides_1d = 1
    
    in_1d = tf.constant(series, dtype=tf.int32)
    
    #in_1d = tf.constant(ones_1d, dtype=tf.float32)
    filter_1d = tf.constant(weight_1d, dtype=tf.int32)
    
    in_width = int(in_1d.shape[0])
    filter_width = int(filter_1d.shape[0])
    
    input_1d   = tf.reshape(in_1d, [1, in_width, 1])
    kernel_1d = tf.reshape(filter_1d, [filter_width, 1, 1])
    output_1d = tf.divide(tf.squeeze(tf.nn.conv1d(input_1d, kernel_1d, strides_1d, padding='SAME')),mat_days)
    return output_1d


@tf.function
def mat_add_2d(series,mat_days):
    
    weight_2d = np.ones((1,mat_days))
    strides_2d = [1, 1, 1, 1]
    
    in_2d = series #tf.constant(series, dtype=tf.float32)
    filter_2d = tf.constant(weight_2d, dtype=tf.int32)
    
    in_width = int(in_2d.shape[1])
    in_height = int(in_2d.shape[0])
    
    filter_width = int(filter_2d.shape[1])
    filter_height = int(filter_2d.shape[0])
    
    input_2d   = tf.reshape(in_2d, [1, in_height, in_width, 1])
    kernel_2d = tf.reshape(filter_2d, [filter_height, filter_width, 1, 1])

    output_2d = tf.cast(tf.divide(tf.squeeze(tf.nn.conv2d(input_2d, kernel_2d, strides=strides_2d, padding='SAME')),mat_days),dtype=tf.float32)
    return output_2d



# tf.random.set_seed(5)
# series=tf.random.uniform(shape=[826], maxval=10, dtype=tf.int32, seed=10)

# mat_days=28
# print("series=\n",series)

# output_1d=mat_add_1d(series,mat_days)
# print("output",output_1d)


tf.random.set_seed(42)
series=tf.constant(tf.random.uniform(shape=[3,826], maxval=100, dtype=tf.int32, seed=10))

mat_days=28  #tf.constant(28,dtype=tf.int32)
print("series=\n",series)

output_2d=mat_add_2d(series,mat_days)
print("output",output_2d)

#series=tf.random.uniform(shape=[3,826], maxval=2, dtype=tf.int32, seed=10)

#mat_days=28
#print("series=\n",series)

#output_2d=mat_add_2d(series,mat_days)
#print("output",output_2d)


#squares = map_fn(lambda x: x * x, elems)
#result=mat_add(series,mat_days)
#print(result)

# r=runningMeanFast(series,3)
# print("r=\n",r,r.shape)