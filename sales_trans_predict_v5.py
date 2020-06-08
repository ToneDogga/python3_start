#  sales_trans predict v1 by Anthony Paech written 25/5/20
# A simple, efficient sales analysis tool
# uses only TF2 functions for speed
# simple intended to test use of TFRecords, TF functions






# sales trans lib contains the sales trans class
# contains
# load - loads the salestrans files from CSV or excel into TFRecords
# query - loads the queryfile from excel and creates a plot dictionary of each query
# preprocess - preprocesses the TFRecords from sales trans.  Applies a MAT's and updates the query dictionary
#                 save the plot dictionary and the queries
# loop through each query 
#   batch - create X batches
#   create Y - create Y batches from X batches
#   train - apply the batches to model and save each model
#  
#   predict - load model and predict into plot dictionary
#
#   results - plot the plot dictionary and send each prediction to excel by month






import sales_trans_lib_v5

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# TensorFlow ≥2.0 is required
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

tf.config.experimental_run_functions_eagerly(False)   #True)   # false

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

#tf.autograph.set_verbosity(3, True)




from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"

#if not tf.config.list_physical_devices('GPU'):
#    print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
#    if IS_COLAB:
#        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")


# Disable all GPUS 
#tf.config.set_visible_devices([], 'GPU') 



 #visible_devices = tf.config.get_visible_devices() 
# for device in visible_devices: 
#     print(device)
#     assert device.device_type != 'GPU' 

#tf.config.set_visible_devices([], 'GPU') 
#tf.config.set_visible_devices(visible_devices, 'GPU') 


#import keras.backend as K


# # Common imports
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
# #import random
import datetime as dt
import gc
import multiprocessing

from numba import cuda
# ""
import collections
from collections import defaultdict
# from datetime import datetime
# #import SCBS0 as c


# # to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# # To plot pretty figures
# #%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_begin(self, logs=None):
    print('Training:  begins at {}'.format(dt.datetime.now().time()))

  def on_train_end(self, logs=None):
    print('Training:  ends at {}'.format(dt.datetime.now().time()))

  def on_predict_begin(self, logs=None):
    print('Predicting: begins at {}'.format(dt.datetime.now().time()))

  def on_predict_end(self, logs=None):
    print('Predicting: ends at {}'.format(dt.datetime.now().time()))



class MCDropout(keras.layers.Dropout):
     def call(self,inputs):
        return super().call(inputs,training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self,inputs):
        return super().call(inputs,training=True)


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


   
 
def plot_learning_curves(loss, val_loss,epochs,title):
    ax = plt.gca()
    ax.set_yscale('log')
    if np.min(loss)>10:
        lift=10
    else:
        lift=1

    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
#    plt.axis([1, epochs+1, 0, np.max(loss[1:])])
  #  plt.axis([1, epochs+1, np.min(loss), np.max(loss)])
    plt.axis([1, epochs+1, np.min(loss)-lift, np.max(loss)])

    plt.legend(fontsize=14)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)






###########################################3





def main():
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

    st=sales_trans_lib_v5.salestrans()   # instantiate the sales trans class

    
    print("\n\nSales Crystal Ball Stack2 : TF2 Salestrans predict - By Anthony Paech 25/5/20")
    print("=============================================================================\n")       
    
    print("Python version:",sys.version)
    print("\ntensorflow:",tf.__version__)
    print("eager exec:",tf.executing_eagerly())

    print("keras:",keras.__version__)
    print("numpy:",np.__version__)
    print("pandas:",pd.__version__)
    print("matplotlib:",mpl.__version__)
    print("salestranslib:",st.__version__)
 #   print("sklearn:",sklearn.__version__)
   
    print("\nnumber of cpus : ", multiprocessing.cpu_count())

    visible_devices = tf.config.get_visible_devices('GPU') 

    print("tf.config.get_visible_devices('GPU'):",visible_devices)


    print("\n============================================================================\n")       


       
    np.random.seed(42)
    tf.random.set_seed(42)
           
 

    answer="y"
    answer=input("Load salestrans?")
    if answer=="y":
        sales_df=st.load_sales(st.filenames)  # filenames is a list of xlsx files to load and sort by date
      #  sales_df=st.preprocess_sales(sales_df)
     
       # print("start sales dataframe=\n",sales_df)    # pandas dataframe

    # =============================================================================
    #  
    # 
    # # the plot dictionary is the holding area of all data
    # # it has a 3-tuple for a key
    # 
    # first is query name
    # second is 0= originsal data, 1 = actual query don't predict or  plot, 2 = plot actual mat, 3 = plot prediction, 4 = plot prediction with error bar
    # third is the start point
    # fourth is the plot number
    #
    # the value is a 1D Tensor except at the start where sales_df is a pandas dataframe
    #     
    # =============================================================================
          

    
    
        plot_dict=dict({('loaded_dataframe',0,0,0) : sales_df})
        st.save_plot_dict(plot_dict,st.plot_dict_filename)
    #    plot_dict[('loaded_model',0,0,0)]=
    #    st.save_plot_dict(plot_dict,st.plot_dict_filename)
        # model_filename="SCBS_model_"+str(qnames[query_number])+".h5"
        # print("\nsave model '",model_filename,"'")
        # model.save(model_filename, include_optimizer=True)
      
        # model_filename_list.append(model_filename)   

    
 
    else:
        
        plot_dict=st.load_plot_dict(st.plot_dict_filename)
     #   plot_dict=st.empty_plot_dict_except_loaded(plot_dict)
        sales_df=plot_dict[('loaded_dataframe',0,0,0)] 
        plot_dict=dict({('loaded_dataframe',0,0,0) : sales_df})    # clear out plot_dict mats and predictions
  #      print("\nsave model\n")
  #       model.save("GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
  
    #    for key in plot_dict.copy():
    #        if key[1]==2 | key[1]==3:
    #          del plot_dict[key]
              



    plot_dict=st.query_sales(sales_df,st.queryfilename,plot_dict)  
    
    del sales_df
    gc.collect()
    
    st.save_plot_dict(plot_dict,st.plot_dict_filename)
#    plot_dict=st.remove_key_from_dict(plot_dict,('loaded_dataframe',0,0))   # to save memory  

 #   print("\nplot_dict=\n",plot_dict.keys())
#    start_dict_keys=plot_dict.keys().copy()
#    print("start dict keys",start_dict_keys)
    
    plot_dict=st.load_plot_dict(st.plot_dict_filename)
    dct=sorted(plot_dict.items(), key=lambda x: x[0][3])
    plot_dict = collections.OrderedDict(dct)
    print("\nQueries created:\n",list(plot_dict.keys()))

 #   for k in plot_dict.keys():
    
    for k in plot_dict.copy():
      #  series=plot_dict[k]
        query_name=k[0]
        plot_number=k[3]
        if k[1]==2:
            print("\nQuery name:",query_name)  #," : ",plot_dict[k][0,-10:])
        #    batches=st.build_all_possible_batches_from_series(plot_dict[k],st.batch_length*2+1)
         #   print("all batches shape=",batches.shape)
            X,Y=st.create_X_and_Y_batches(plot_dict[k],st.batch_length,st.no_of_batches)
           # X,Y=st.create_X_and_Y_batches(plot_dict[k],st.batch_length,st.no_of_batches)
            dataset=tf.data.Dataset.from_tensor_slices((X,Y)).cache().repeat(st.no_of_repeats)
     #       dataset=tf.data.Dataset.from_tensor_slices((X[:,:,:1],Y)).cache().repeat(st.no_of_repeats)

            # dataset=dataset.cache().repeat(st.no_of_repeats)
 
    #  #   dataset=dataset.map(preprocess,num_parallel_calls=None)
        #    dataset=dataset.cache() 
            dataset=dataset.shuffle(buffer_size=st.no_of_batches+1,seed=42)
         #   shapes = (tf.TensorShape([None,1]),tf.TensorShape([None,st.batch_length]))
            shapes = (tf.TensorShape([None,1]),tf.TensorShape([None,st.batch_length]))

#            train_set = dataset.padded_batch(1,padded_shapes=shapes, padding_values=(0,0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))
#            valid_set = dataset.padded_batch(1,padded_shapes=shapes, padding_values=(0,0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))

         #   train_set = dataset.padded_batch(1,padded_shapes=shapes).prefetch(1)   #, padding_values=(None, None))
         #   valid_set = dataset.padded_batch(1,padded_shapes=shapes).prefetch(1)   #, padding_values=(None, None))




        #    train_set = dataset.padded_batch(1,padded_shapes=([st.batch_length,st.batch_length], [st.batch_length,st.batch_length]), padding_values=(-1, 0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))
        #    valid_set = dataset.padded_batch(1,padded_shapes=([st.batch_length,st.batch_length], [st.batch_length,st.batch_length]), padding_values=(-1, 0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))

        #    train_set=dataset.batch(1,drop_remainder=True).prefetch(1)
        #    valid_set=dataset.batch(1,drop_remainder=True).prefetch(1)
            
            
            train_set=dataset.batch(1).prefetch(1)
            valid_set=dataset.batch(1).prefetch(1)
        
     #####################################3
     # model goes here
     
            model = keras.models.Sequential([
          #     keras.layers.Conv1D(filters=st.batch_length,kernel_size=4, strides=1, padding='same', input_shape=[None, 1]),  #st.batch_length]), 
          #     keras.layers.BatchNormalization(),
               keras.layers.GRU(st.neurons, return_sequences=True, input_shape=[None, 1]), #st.batch_length]),
               keras.layers.BatchNormalization(),
               keras.layers.GRU(st.neurons, return_sequences=True),
               keras.layers.AlphaDropout(rate=st.dropout_rate),
               keras.layers.BatchNormalization(),
               keras.layers.TimeDistributed(keras.layers.Dense(st.batch_length))
            ])
        
            model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
           
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=st.patience),MyCustomCallback()]
           
            history = model.fit(train_set ,epochs=st.epochs,
                               validation_data=(valid_set), callbacks=callbacks)

#            history = model.fit(train_set,  steps_per_epoch=st.steps_per_epoch ,epochs=st.epochs,
#                               validation_data=(valid_set))
      
        #      history = model.fit_generator(X_train, Y_train, epochs=st.epochs,
      #                         validation_data=(X_valid, Y_valid))
           
           
     #       print("\nsave model\n")
            model.save(st.output_dir+query_name+":GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
              
         #   model.summary()
           
            plot_learning_curves(history.history["loss"], history.history["val_loss"],st.epochs,"GRU and dropout:"+str(query_name))
            st.save_fig("GRU and dropout learning curve_"+query_name,st.images_path)
        
            plt.show()
     #############################################################
# =============================================================================
#             #  Wavenet     
#            
#             model = keras.models.Sequential()
#             model.add(keras.layers.InputLayer(input_shape=[None, 1]))
#             for rate in (1, 2, 4, 8) * 2:
#                 model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
#                                               activation="relu", dilation_rate=rate))
#             model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
#             model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
#             history = model.fit(train_set, epochs=st.epochs,
#                                 validation_data=(valid_set))
#                  
#      
#         
#            
#         ########################################3
# =============================================================================
        #  send predictions to plot_dict and then excel    
            
        #    tf.keras.backend.clear_session()
            # gc.collect()
             
            print("\nPredicting....",query_name)
         #   series=series[...,tf.newaxis]
 #           new_prediction,new_stddev=st.predict_series(model,plot_dict[k][:,st.start_point:st.end_point+1][...,tf.newaxis])
          #  new_prediction,new_stddev=st.predict_series(model,plot_dict[k][...,tf.newaxis])
            new_prediction,new_stddev=st.simple_predict(model,plot_dict[k][...,tf.newaxis])

        #    print("new predictopn=",new_prediction,new_prediction.shape)
      #      print("predict ahead=",predict_ahead)
            
            plot_dict=st.append_plot_dict(plot_dict,query_name,new_prediction,new_stddev,plot_number)  
     #       plot_dict=st.append_plot_dict(plot_dict,query_name,new_prediction,plot_number)  
        
            st.save_plot_dict(plot_dict,st.output_dir+st.plot_dict_filename)
 
            
         
     #   gc.collect()
     #   tf.keras.backend.clear_session()
        #cuda.select_device(0)
    #cuda.close()
    
    

    st.save_plot_dict(plot_dict,st.plot_dict_filename)

  #  print("purging plot_dict of non plottable data")
    for key in plot_dict.copy():
        if ((key[1]==1) | (key[1]==0)):
              del plot_dict[key]
    
  #  print("plot dict purged")        
 #   for key in plot_dict.keys():
 #       print(key,plot_dict[key].shape)
     
        
    # sort by plot_number  
    dct=sorted(plot_dict.items(), key=lambda x: x[0][3])
    plot_dict = collections.OrderedDict(dct)
 #   print("\n2sorted plot_dict after prediction",plot_dict.keys())
  
    new_plot_df,new_column_names=st.build_final_plot_df(plot_dict)
  
    print("Plotting plot_dict...")
  #  print("new_plot df=\n",new_plot_df.columns,"->",new_column_names,new_plot_df.shape)

    st.plot_new_plot_df(new_plot_df)
    new_plot_df=st.simplify_col_names(new_plot_df,new_column_names)
#    new_plot_df=st.clean_up_col_names(new_plot_df)
       #    plot_dict[key[0]] = plot_dict.pop(key)

    #plot_df=pd.DataFrame.from_dict(plot_dict,orient='index',dtype=np.int32)
  
    print("\nwrite predictions to sales_prediction(.....).CSV file....")
   
    print("Saving pickled final table - final_series_table.pkl",new_plot_df.shape)
    
        #    series_table=series_table.T       
    pd.to_pickle(new_plot_df,"final_series_tables.pkl")
 
    new_plot_df=st.clean_up_col_names(new_plot_df)  
    
    forecast_table = new_plot_df.resample('M', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)

    forecast_table.index=forecast_table.index.strftime("%Y-%m-%d")

    forecast_table.to_excel(st.output_dir+"SCBS2_forecast_table.xlsx") 
        
             
  #  print("\n\npredict module finish\n\n")
 
    print("\n\nFinished.")
    gc.collect()      
        
        
    return


if __name__ == '__main__':
    main()

        
          
          
          

