# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

tf.config.experimental_run_functions_eagerly(False)
#tf.config.run_functions_eagerly(False)   #True)
from tensorflow import keras

import numpy as np
import timeit
import csv

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time
import itertools
from collections import defaultdict
import pickle


# # to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# # To plot pretty figures
# #%matplotlib inline
import matplotlib as mpl
#import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)



# def runningMeanFast(x, N):
# #    return tf.convolve(x, np.ones((N,))/N)[(N-1):]
#     return tf.nn.convolution(x, np.ones((N,))/N)[(N-1):]



# def mat_add(series,mat_days):
#     result=tf.Variable(tf.zeros((series.shape[0],series.shape[1]),dtype=tf.int32))
#  #   rm=np.rolling_apply(series, lambda x: np.mean(x), window=10)
#    # squares = tf.map_fn(lambda x: x * x, series)
#     return squares



#ones_1d = np.ones(4)
@tf.function
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


# tf.random.set_seed(42)
# #series=tf.constant(tf.random.uniform(shape=[1,826], maxval=100, dtype=tf.int32, seed=10))
# series=tf.Variable(tf.random.uniform(shape=[1,826], maxval=100, dtype=tf.int32, seed=10))

# # mat_days=28  #tf.constant(28,dtype=tf.int32)
# # print("series=\n",series)

# # output_2d=mat_add_2d(series,mat_days)
# # print("output",output_2d)

# #series=tf.random.uniform(shape=[3,826], maxval=2, dtype=tf.int32, seed=10)

# #mat_days=28
# #print("series=\n",series)

# #output_2d=mat_add_2d(series,mat_days)
# #print("output",output_2d)


# #squares = map_fn(lambda x: x * x, elems)
# #result=mat_add(series,mat_days)
# #print(result)

# # r=runningMeanFast(series,3)
# # print("r=\n",r,r.shape)

# print(series.shape)




# np.random.seed(42)

# n_steps = 50
# series = generate_time_series(10000, n_steps + 10)
# X_train = series[:7000, :n_steps]
# X_valid = series[7000:9000, :n_steps]
# X_test = series[9000:, :n_steps]
# Y = np.empty((10000, n_steps, 10))
# for step_ahead in range(1, 10 + 1):
#     Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
# Y_train = Y[:7000]
# Y_valid = Y[7000:9000]
# Y_test = Y[9000:]

# #@tf.function
# def create_Y(series,X,batch_length):
#     print("series shape=",series.shape)
#   #  tf.random.set_seed(42)
    
#     Y=tf.Variable(tf.zeros((no_of_batches,batch_length,batch_length),tf.int32))
#     #Y=tf.TensorArray(tf.zeros((no_of_batches,batch_length,batch_length),tf.int32),dynamic_size=True)
#     #Y = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

#  #   for step_ahead in tf.range(1, batch_length + 1):
#      #   print("v=",series[..., step_ahead:step_ahead + n_steps],series[..., step_ahead:step_ahead + n_steps].shape)
#     #    Y= Y[step_ahead - 1].assign(series[step_ahead:step_ahead + batch_length])
#     Y=[tf.stack(tf.broadcast_to(series[0,step_ahead:step_ahead + batch_length],[no_of_batches,batch_length]),axis=-1) for step_ahead in tf.range(1, batch_length + 1)]

# #    Y=[tf.stack(tf.broadcast_to(series[0,step_ahead:step_ahead + batch_length],[no_of_batches,batch_length]),axis=-1) for step_ahead in tf.range(1, batch_length + 1)]
#   #  print("Y1=",Y)
#  #   Y=tf.transpose(Y)

#   #  Z= tf.map_fn(lambda i: i ** 2 if i > 0 else i, x)


#     return Y
# #    return tf.cast(Y,dtype=tf.int32)
# #aa=tf.Variable(tf.zeros(3, tf.int32))
# #aa=aa[2].assign(1)
# x = tf.constant([1, 2, 3])
# y = tf.broadcast_to(x, [3, 3])
# print(y)


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])



class MCDropout(keras.layers.Dropout):
     def call(self,inputs):
        return super().call(inputs,training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self,inputs):
        return super().call(inputs,training=True)



    
 
def plot_learning_curves(loss, val_loss,epochs,title):
    ax = plt.gca()
    ax.set_yscale('log')

    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
#    plt.axis([1, epochs+1, 0, np.max(loss[1:])])
    plt.axis([1, epochs+1, 0, np.max(loss)])

    plt.legend(fontsize=14)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)




     
# #  @tf.function
# def create_Y2(series,no_of_batches,batch_length):
#      Y = np.empty((no_of_batches, batch_length, batch_length))
#      for step_ahead in range(1, batch_length + 1):
#          Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + batch_length]

#      return tf.convert_to_tensor(Y,tf.int32) 

#  ###########################################3
 
 
 
# class SeriesDataset(tf.data.Dataset):
#     def _generator(num_samples):
#         # Opening the file
#         time.sleep(0.03)
        
#         for sample_idx in range(num_samples):
#             # Reading data (line, record) from the file
#             time.sleep(0.015)
            
#             yield (sample_idx,)
    
#     def __new__(cls, num_samples=3):
#         return tf.data.Dataset.from_generator(
#             cls._generator,
#             output_types=tf.dtypes.int64,
#             output_shapes=(1,),
#             args=(num_samples,)
#         )


# def benchmark(dataset, num_epochs=2):
#     start_time = time.perf_counter()
#     for epoch_num in range(num_epochs):
#         for sample in dataset:
#             # Performing a training step
#             time.sleep(0.01)
#     tf.print("Execution time:", time.perf_counter() - start_time)



# def mapped_function(s):
#     # Do some hard pre-processing
#     tf.py_function(lambda: time.sleep(0.03), [], ())
#     print(s)
#     return s



def load_plot_dict():  #:
    with open('plot_dict.pkl',"rb") as f:
        return pickle.load(f)


# #
# #


# ##############################################



# #fast_dataset = tf.data.Dataset.range(10000)

# def fast_benchmark(dataset, num_epochs=2):
#     start_time = time.perf_counter()
#     for _ in tf.data.Dataset.range(num_epochs):
#         for _ in dataset:
#             pass
#     tf.print("Execution time:", time.perf_counter() - start_time)
    
# def increment(x):
#     return x+1

####################################33


# def generator(batch_size):
#     for i in range(batch_size):
#       yield 2*i
 
###################################################

# def generator():
#  #   start=np.random.randint(0,series.shape[1]-batch_size-1)
#  #   finish=start+batch_size     
#  #   X_chunks=series[start:finish]
#  #   Y_chunks=series[start:finish]

#     for i, j in zip(X_chunks, Y_chunks):
#         yield i, j




 #       start=np.random.randint(0,series.shape[1]-batch_size)
#        finish=start+batch_size
        
      #   Y_chunks=series[start:finish]

   #     Y_chunks=np.random.choice(series,size=32,replace=False)

#        Y_chunks=series.random.random_sample((1,32))
        
        # X_chunks = list(np.array_split(series, series.shape[1]/32,axis=1))
        # Y_chunks = list(np.array_split(series, series.shape[1]/32,axis=1))

        # train_dataset = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32))
        # train_dataset = train_dataset.shuffle(30).batch(1).repeat().prefetch(1)     
        # iterator = iter(train_dataset)


# #ones_1d = np.ones(4)
# @tf.function
# def mat_add_1d(series,mat_days):
#     weight_1d = np.ones(mat_days)
#     strides_1d = 1
    
#     in_1d = tf.constant(series, dtype=tf.int32)
    
#     #in_1d = tf.constant(ones_1d, dtype=tf.float32)
#     filter_1d = tf.constant(weight_1d, dtype=tf.int32)
    
#     in_width = int(in_1d.shape[0])
#     filter_width = int(filter_1d.shape[0])
    
#     input_1d   = tf.reshape(in_1d, [1, in_width, 1])
#     kernel_1d = tf.reshape(filter_1d, [filter_width, 1, 1])
#     output_1d = tf.divide(tf.squeeze(tf.nn.conv1d(input_1d, kernel_1d, strides_1d, padding='SAME')),mat_days)
#     return output_1d

#@tf.function
def build_Y(series,unique_batches,batch_length):
    Y = np.empty((unique_batches, batch_length, batch_length),dtype=np.int32)
    for p in tf.range(batch_length,unique_batches):
        X_batch=series[0,p:p+batch_length+batch_length+1] 
        for step_ahead in tf.range(1, batch_length + 1):
            Y[...,step_ahead - 1] = X_batch[..., step_ahead:step_ahead + batch_length]
    #    print(Y)    
    return Y   #tf.convert_to_tensor(Y,tf.int32) 


# def get_X(series, batch_length, unique_batches):
#      return tf.convert_to_tensor([tf.stack((series[0,p:p+batch_length])) for p in tf.range(batch_length,unique_batches)],dtype=tf.int32) #.to_numpy()



#@tf.function
def create_X_and_Y_batches(series,batch_length):
    unique_batches=series.shape[1]-batch_length-batch_length-1
    X_batches=tf.convert_to_tensor([tf.stack((series[0,p:p+batch_length])) for p in tf.range(batch_length,unique_batches)],dtype=tf.int32) #.to_numpy()
   # tf.print("XB",X_batches,len(X_batches))   #,X_batches.shape)
    X_batches=X_batches[...,tf.newaxis]   
 #   print("X shape",X_batches.shape)

    Y_batches=tf.convert_to_tensor(build_Y(series,unique_batches-batch_length,batch_length),dtype=tf.int32)
    
  #  print("Y shape",Y_batches.shape)
 
    dataset=tf.data.Dataset.from_tensor_slices((tf.repeat(X_batches,batch_length,axis=2),Y_batches)).repeat(20)

 #   dataset=dataset.map(preprocess,num_parallel_calls=None)
         
    dataset=dataset.shuffle(buffer_size=1000,seed=42)
    return dataset.batch(1).prefetch(1)



#def preprocess(line):
    # turn a tensor of shape (1,series length)
    # into every batch of length batch_size   into tensor shape[1,batch_size,1]  }
    
  #  batches=tf.data.Dataset.from_tensor_slices(line).batch(10,drop_remainder=True)
#    tf.print("batches",batches)
#     weight_1d = np.ones(batch_size)
#     strides_1d = 1
    
#     in_1d = line #tf.reshape(line,[1,-1])  #tf.constant(line, dtype=tf.int32)
#     print("line =",line,line.shape,"in_1d=",in_1d)
#     #in_1d = tf.constant(ones_1d, dtype=tf.float32)
#     filter_1d = tf.constant(weight_1d, dtype=tf.int32)
    
#     in_width = int(in_1d.shape[0])
#     filter_width = int(filter_1d.shape[0])
#     print("in width",in_width,"filter width",filter_width)
    
# #    input_1d   = tf.reshape(in_1d, [1, in_width, 1])
#     input_1d   = tf.reshape(in_1d, [1,in_width, 1])

#     print("0batch_seize",batch_size,"input 1D",input_1d,"filter_1d",filter_1d)
# #    kernel_1d = tf.reshape(filter_1d, [filter_width, 1, 1])

#     kernel_1d = tf.reshape(filter_1d, [filter_width,1,1])
#     print("1batch_seize",batch_size,"input 1D",input_1d,"filter_1d",filter_1d,"kernel 1d",kernel_1d)

#     output_1d = tf.nn.conv1d(input_1d, kernel_1d, strides_1d, padding='SAME')   # tf.squeeze
#     print("2batch_seize",batch_size,"input 1D",input_1d,"filter_1d",filter_1d,"kernel 1d",kernel_1d,"outp[ut 1d",output_1d)
#     return output_1d


 #   return line
    




batch_length=16

plot_dict=load_plot_dict()
for k in plot_dict.keys():
    if k[1]==2:
     #   print(('rep36_all@28u:mt', 2, 32)," : ",plot_dict[('rep36_all@28u:mt', 2, 32)][:10])
        print("\n",k[0])  #," : ",plot_dict[k][0,-10:])
#        dataset = tf.data.Dataset.from_tensor_slices(plot_dict[k])

        train_set=create_X_and_Y_batches(plot_dict[k],batch_length)
        valid_set=create_X_and_Y_batches(plot_dict[k],batch_length)
        test_set=create_X_and_Y_batches(plot_dict[k],batch_length)

 
     #   for _ in range(6):
         #   tf.print(iterator.get_next())  #,i[0].shape)      
       # tf.print(train_set.shape)   #.get_next())
            
            
                    
#          #       iter = dataset.make_one_shot_iterator()
#         tf.print(iterator.get_next()[0][0])   # make a simple model
        
       
        #####################################3
# model goes here
#
        epochs=12
        model = keras.models.Sequential([
          keras.layers.GRU(batch_length, return_sequences=True, input_shape=[None, batch_length]),
          keras.layers.BatchNormalization(),
          keras.layers.GRU(batch_length, return_sequences=True),
          keras.layers.AlphaDropout(rate=0.2),
          keras.layers.BatchNormalization(),
          keras.layers.TimeDistributed(keras.layers.Dense(batch_length))
        ])
   
        model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
      
    #    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=st.patience),MyCustomCallback()]
      
        history = model.fit(train_set, epochs=epochs,
                          validation_data=(valid_set))
    #    history = model.fit_generator(iterator.get_next()[0],iterator.get_next()[1], epochs=3)    #,
    #                  #    validation_data=(X_valid, Y_valid))
      
      
    #    print("\nsave model\n")
    #    model.save("GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
         
    #   model.summary()
      
        plot_learning_curves(history.history["loss"], history.history["val_loss"],epochs,"GRU and dropout")
    #    save_fig("GRU and dropout learning curve",st.images_path)
   
        plt.show()
   
   
          
        ########################################3

       
        
        
        
        
        # net = tf.keras.layers.Dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
        # net = tf.keras.layers.Dense(net, 8, activation=tf.tanh)
        # prediction = tf.keras.layers.dense(net, 1, activation=tf.tanh)
        # loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
        # train_op = tf.train.AdamOptimizer().minimize(loss)
        
            
# with open('dict_output.csv', 'wb') as outfile:
#     listWriter = csv.DictWriter(
#        outfile,
#        fieldnames=plot_dict["date"],
#        delimiter=',',
#        quotechar='|',
#        quoting=csv.QUOTE_MINIMAL
#     )


#

# print("fast dataset generator")
# fast_dataset = tf.data.Dataset.from_tensor_slices(series)

# # benchmark(
# #     SeriesDataset()
# #     .map(
# #         mapped_function,
# #         num_parallel_calls=tf.data.experimental.AUTOTUNE
# #     )
# # )


# X=fast_dataset.batch(32).map(increment)
#     # Apply function on a batch of items
#     # The tf.Tensor.__add__ method already handle batches
  
# for i in X:
#     print("X{i}=",i)





# source data - numpy array
#data = np.arange(10)
# create a dataset from numpy array
        # dataset = tf.data.Dataset.from_tensor_slices(series).batch(32).prefetch(1)
        # iterator = iter(dataset)
        # for x in range(20):
        #     tf.print(iterator.get_next())


       
     #   dataset = tf.data.Dataset.from_generator(generator, (tf.int32))
        
     #   for X in dataset:
     #       print("x=",X)


# fast_benchmark(
#     fast_dataset
#     .batch(256)
#     # Apply function on a batch of items
#     # The tf.Tensor.__add__ method already handle batches
#     .map(increment)
# )






# with open('dict_output.csv', 'wb') as outfile:
#     listWriter = csv.DictWriter(
#        outfile,
#        fieldnames=itemDict[itemDict.keys()[0]].keys(),
#        delimiter=',',
#        quotechar='|',
#        quoting=csv.QUOTE_MINIMAL
#     )


# tf.random.set_seed(42)
# #series=tf.constant(tf.random.uniform(shape=[1,826], maxval=100, dtype=tf.int32, seed=10))
# series=tf.Variable(tf.random.uniform(shape=[1,826], maxval=100, dtype=tf.int32, seed=10))

# # mat_days=28  #tf.constant(28,dtype=tf.int32)
# # print("series=\n",series)

# # output_2d=mat_add_2d(series,mat_days)
# # print("output",output_2d)

# #series=tf.random.uniform(shape=[3,826], maxval=2, dtype=tf.int32, seed=10)

# #mat_days=28
# #print("series=\n",series)

# #output_2d=mat_add_2d(series,mat_days)
# #print("output",output_2d)


# #squares = map_fn(lambda x: x * x, elems)
# #result=mat_add(series,mat_days)
# #print(result)

# # r=runningMeanFast(series,3)
# # print("r=\n",r,r.shape)

# print(series.shape)




# print("sers",series.shape)

# tf_Y=create_Y(series,10,100)  #tf.constant(8,shape=(),dtype=tf.int32),tf.constant(7,shape=(),dtype=tf.int32))
# print("tf_Y=\n",tf_Y)


# Y=create_Y2(series,10,100)  #tf.constant(8,shape=(),dtype=tf.int32),tf.constant(7,shape=(),dtype=tf.int32))
# print("Y=\n",Y)

# # create a random vector of shape (100,2)
# x = np.random.sample((100,2))
# # make a dataset from a numpy array
# dataset = tf.data.Dataset.from_tensor_slices(x)
# # np.random.seed(42)
# features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
# dataset = tf.data.Dataset.from_tensor_slices((features,labels))
# # n_steps = 6
# # series=series.numpy()
# # #series = generate_time_series(10000, n_steps + 10)
# # #X_train = series[:7000, :n_steps]
# # #X_valid = series[7000:9000, :n_steps]
# # #X_test = series[9000:, :n_steps]
# # Y = np.empty((7, n_steps, n_steps))
# # print(Y,Y.shape)
    
# # using a tensor
# dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))


# # from generator
# sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])

# def generator():
#     for el in sequence:
#         yield el
        
# dataset = tf.data.Dataset().batch(1).from_generator(generator, output_types= tf.int32, output_shapes=(tf.TensorShape([None, 1])))

# iter = dataset.make_initializable_iterator()
# el = iter.get_next()
# print(el)

# # with make_reader(output_url, num_epochs=num_epochs) as reader:
#     dataset = make_petastorm_dataset(reader)
#     for data in dataset:
#         print(data.id)

########################################333


# # BATCHING
# BATCH_SIZE = 4
# x = np.random.sample((100,2))
# # make a dataset from a numpy array
# dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)
# iter = tf.data.Dataset.iter()
# el = iter.get_next()
# print(el)

# #########################################################################3-2021

# class TimeMeasuredDataset(tf.data.Dataset):
#     # OUTPUT: (steps, timings, counters)
#     OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)
#     OUTPUT_SHAPES = ((2, 1), (2, 2), (2, 3))
    
#     _INSTANCES_COUNTER = itertools.count()  # Number of datasets generated
#     _EPOCHS_COUNTER = defaultdict(itertools.count)  # Number of epochs done for each dataset
    
#     def _generator(instance_idx, num_samples):
#         epoch_idx = next(TimeMeasuredDataset._EPOCHS_COUNTER[instance_idx])
        
#         # Opening the file
#         open_enter = time.perf_counter()
#         time.sleep(0.03)
#         open_elapsed = time.perf_counter() - open_enter
        
#         for sample_idx in range(num_samples):
#             # Reading data (line, record) from the file
#             read_enter = time.perf_counter()
#             time.sleep(0.015)
#             read_elapsed = time.perf_counter() - read_enter
            
#             yield (
#                 [("Open",), ("Read",)],
#                 [(open_enter, open_elapsed), (read_enter, read_elapsed)],
#                 [(instance_idx, epoch_idx, -1), (instance_idx, epoch_idx, sample_idx)]
#             )
#             open_enter, open_elapsed = -1., -1.  # Negative values will be filtered
            
    
#     def __new__(cls, num_samples=3):
#         return tf.data.Dataset.from_generator(
#             cls._generator,
#             output_types=cls.OUTPUT_TYPES,
#             output_shapes=cls.OUTPUT_SHAPES,
#             args=(next(cls._INSTANCES_COUNTER), num_samples)
#         )
    
    
    
    
# ##############################################################333
# def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):
#     # Remove invalid entries (negative times, or empty steps) from the timelines
#     invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:,0]
#     steps = timeline['steps'][invalid_mask].numpy()
#     times = timeline['times'][invalid_mask].numpy()
#     values = timeline['values'][invalid_mask].numpy()
    
#     # Get a set of different steps, ordered by the first time they are encountered
#     step_ids, indices = np.stack(np.unique(steps, return_index=True))
#     step_ids = step_ids[np.argsort(indices)]

#     # Shift the starting time to 0 and compute the maximal time value
#     min_time = times[:,0].min()
#     times[:,0] = (times[:,0] - min_time)
#     end = max(width, (times[:,0]+times[:,1]).max() + 0.01)
    
#     cmap = mpl.cm.get_cmap("plasma")
#     plt.close()
#     fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
#     fig.suptitle(title)
#     fig.set_size_inches(17.0, len(step_ids))
#     plt.xlim(-0.01, end)
    
#     for i, step in enumerate(step_ids):
#         step_name = step.decode()
#         ax = axs[i]
#         ax.set_ylabel(step_name)
#         ax.set_ylim(0, 1)
#         ax.set_yticks([])
#         ax.set_xlabel("time (s)")
#         ax.set_xticklabels([])
#         ax.grid(which="both", axis="x", color="k", linestyle=":")
        
#         # Get timings and annotation for the given step
#         entries_mask = np.squeeze(steps==step)
#         serie = np.unique(times[entries_mask], axis=0)
#         annotations = values[entries_mask]
        
#         ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
#         if annotate:
#             for j, (start, width) in enumerate(serie):
#                 annotation = "\n".join([f"{l}: {v}" for l,v in zip(("i", "e", "s"), annotations[j])])
#                 ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
#                         horizontalalignment='left', verticalalignment='center')
#     if save:
#         plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")

# ############################################################

# def timelined_benchmark(dataset, num_epochs=2):
#     # Initialize accumulators
#     steps_acc = tf.zeros([0, 1], dtype=tf.dtypes.string)
#     times_acc = tf.zeros([0, 2], dtype=tf.dtypes.float32)
#     values_acc = tf.zeros([0, 3], dtype=tf.dtypes.int32)
    
#     start_time = time.perf_counter()
#     for epoch_num in range(num_epochs):
#         epoch_enter = time.perf_counter()
#         for (steps, times, values) in dataset:
#             # Record dataset preparation informations
#             steps_acc = tf.concat((steps_acc, steps), axis=0)
#             times_acc = tf.concat((times_acc, times), axis=0)
#             values_acc = tf.concat((values_acc, values), axis=0)
            
#             # Simulate training time
#             train_enter = time.perf_counter()
#             time.sleep(0.01)
#             train_elapsed = time.perf_counter() - train_enter
            
#             # Record training informations
#             steps_acc = tf.concat((steps_acc, [["Train"]]), axis=0)
#             times_acc = tf.concat((times_acc, [(train_enter, train_elapsed)]), axis=0)
#             values_acc = tf.concat((values_acc, [values[-1]]), axis=0)
        
#         epoch_elapsed = time.perf_counter() - epoch_enter
#         # Record epoch informations
#         steps_acc = tf.concat((steps_acc, [["Epoch"]]), axis=0)
#         times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)
#         values_acc = tf.concat((values_acc, [[-1, epoch_num, -1]]), axis=0)
#         time.sleep(0.001)
    
#     tf.print("Execution time:", time.perf_counter() - start_time)
#     return {"steps": steps_acc, "times": times_acc, "values": values_acc}


# ##########################################################





    
# benchmark(ArtificialDataset())

# benchmark(
#     ArtificialDataset()
#     .prefetch(tf.data.experimental.AUTOTUNE)
# )


# benchmark(
#     tf.data.Dataset.range(2)
#     .interleave(ArtificialDataset)
# )


# benchmark(
#     tf.data.Dataset.range(2)
#     .interleave(
#         ArtificialDataset,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE
#     )
# )


# benchmark(
#     ArtificialDataset()
#     .map(mapped_function)
# )




# benchmark(
#     ArtificialDataset()
#     .map(
#         mapped_function,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE
#     )
# )




# fast_benchmark(
#     fast_dataset
#     # Apply function one item at a time
#     .map(increment)
#     # Batch
#     .batch(256)
# )

# fast_benchmark(
#     fast_dataset
#     .batch(256)
#     # Apply function on a batch of items
#     # The tf.Tensor.__add__ method already handle batches
#     .map(increment)
# )





# # source data - numpy array
# data = np.arange(10)
# # create a dataset from numpy array
# dataset = tf.data.Dataset.from_tensor_slices(data)


# def generator():
#   for i in range(10):
#     yield 2*i
    
# dataset = tf.data.Dataset.from_generator(generator, (tf.int32))

# for X in dataset:
#     print("x=",X)



# datax = np.arange(10,20)
# datay = np.arange(11,21)
# datasetx = tf.data.Dataset.from_tensor_slices(datax)
# datasety = tf.data.Dataset.from_tensor_slices(datay)
# dcombined = tf.data.Dataset.zip((datasetx, datasety)).batch(2)
# iterator = dcombined.make_one_shot_iterator()
# next_ele = iterator.get_next()



# # # using two numpy arrays
# # features, labels = (np.array([np.random.sample((100,2))]), 
# #                     np.array([np.random.sample((100,1))]))


# # dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)


# # iter = dataset.make_one_shot_iterator()
# # x, y = iter.get_next()# make a simple model
# # net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
# # net = tf.layers.dense(net, 8, activation=tf.tanh)
# # prediction = tf.layers.dense(net, 1, activation=tf.tanh)
# # loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
# # train_op = tf.train.AdamOptimizer().minimize(loss)








# # source data - numpy array
# data = np.arange(10)
# # create a dataset from numpy array
# dataset = tf.data.Dataset.from_tensor_slices(data)


# def generator():
#   for i in range(10):
#     yield 2*i
    
# dataset = tf.data.Dataset.from_generator(generator, (tf.int32))

# for X in dataset:
#     print("x=",X)


