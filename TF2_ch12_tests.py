#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:42:57 2020

@author: tonedogga
"""


import tensorflow as tf
#gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import pickle
import os
import sys
import multiprocessing

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.compat.v1.enable_eager_execution()
#print(tf.executing_eagerly())


@tf.function
def tf_build_mini_batches(data_input,no_of_batches,batch_length):   #,start_point,end_point):
    repeats_needed=int(no_of_batches/int(((data_input.shape[1])/batch_length)+1))  #      repeats_needed=int(no_of_batches/(end_point-start_point-start_point-batch_length))
    gridtest=(tf.meshgrid(tf.range(0,batch_length,dtype=tf.int32),tf.range(0,int(((data_input.shape[1])/batch_length)+1),dtype=tf.int32)))   #int((end_point-start_point)/batch_length)+1))
    start_index=tf.random.shuffle(tf.convert_to_tensor(tf.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)))   #[:,:,np.newaxis
    new_batches= tf.random.shuffle(tf.cast(tf.gather(data_input,start_index,axis=1),dtype=tf.float32))
    return new_batches[0][...,tf.newaxis]





print("\nBatch creator\n\n")
print("Python version:",sys.version)
print("tensorflow:",tf.__version__)
#print("keras:",keras.__version__)
print("numpy:",np.__version__)
print("\nnumber of cpus : ", multiprocessing.cpu_count())


# visible_devices = tf.config.get_visible_devices('GPU') 

# print("tf.config.get_visible_devices('GPU'):",visible_devices)
# # else:
# tf.config.set_visible_devices(visible_devices, 'GPU') 
# print("GPUs enabled")



visible_devices = tf.config.get_visible_devices('GPU') 
print("tf.config.get_visible_devices('GPU'):",visible_devices)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors







tf.random.set_seed(43)
print("unpickling '","tables_dict.pkl","'")  
with open("tables_dict.pkl", "rb") as f:
    all_tables = pickle.load(f)

series_table=all_tables[0][1].swapaxes(0,1).to_numpy()
print("series table=",series_table.shape,type(series_table))
  

batches=tf_build_mini_batches(tf.constant(series_table),1000000,34)

#print("2batches shape",batches2.shape)

print(batches[999845:,:,0])


