#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:12:51 2020

@author: tonedogga
"""
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import SCBS0 as c

filename="tables_dict.pkl"



# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow import keras
assert tf.__version__ >= "2.0"

print("\n\nTurn a list of tables into a list of batches - By Anthony Paech 25/2/20")
print("========================================================================")       

print("Python version:",sys.version)
print("\ntensorflow:",tf.__version__)
print("keras:",keras.__version__)
print("sklearn:",sklearn.__version__)


import os
import random
import csv
import joblib
import pickle
from natsort import natsorted
from pickle import dump,load
import datetime as dt
from datetime import date
from datetime import timedelta

#from sklearn.preprocessing import StandardScaler,MinMaxScaler

#import itertools
#from natsort import natsorted
#import import_constants as ic

print("numpy:",np.__version__)
print("pandas:",pd.__version__)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import multiprocessing
print("\nnumber of cpus : ", multiprocessing.cpu_count())


visible_devices = tf.config.get_visible_devices('GPU') 

print("tf.config.get_visible_devices('GPU'):",visible_devices)
# answer=input("Use GPU?")
# if answer =="n":

#     try: 
#       # Disable all GPUS 
#       tf.config.set_visible_devices([], 'GPU') 
#       visible_devices = tf.config.get_visible_devices() 
#       for device in visible_devices: 
#         assert device.device_type != 'GPU' 
#     except: 
#       # Invalid device or cannot modify virtual devices once initialized. 
#       pass 
    
#     #tf.config.set_visible_devices([], 'GPU') 
    
#     print("GPUs disabled")
    
# else:
tf.config.set_visible_devices(visible_devices, 'GPU') 
print("GPUs enabled")
   
    

# if not tf.config.get_visible_devices('GPU'):
# #if not tf.test.is_gpu_available():
#     print("\nNo GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
#   #  if IS_COLAB:
#   #      print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
# else:
#     print("\nSales prediction - GPU detected.")


print("tf.config.get_visible_devices('GPU'):",tf.config.get_visible_devices('GPU'))


# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)


# To plot pretty figures
#%matplotlib inline




# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

#n_steps = 50



def plot_log_learning_curves(title,epochs,loss, val_loss,query_name):
    ax = plt.gca()
    ax.set_yscale('log')
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, np.amin(loss), np.amax(loss)])
    plt.legend(fontsize=11)
    plt.title(title,fontsize=11)
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.grid(True)
    save_fig("log_learning_curve "+str(query_name))
    return


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])



class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_begin(self, logs=None):
    print('Training:  begins at {}'.format(dt.datetime.now().time()))

  def on_train_end(self, logs=None):
    print('Training:  ends at {}'.format(dt.datetime.now().time()))

  def on_predict_begin(self, logs=None):
    print('Predicting: begins at {}'.format(dt.datetime.now().time()))

  def on_predict_end(self, logs=None):
    print('Predicting: ends at {}'.format(dt.datetime.now().time()))



# class MCDropout(keras.layers.Dropout):
#     def call(self,inputs):
#         return super().call(inputs,training=True)


class MCDropout(keras.layers.AlphaDropout):
    def call(self,inputs):
        return super().call(inputs,training=True)












def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


    
 
   

def graph_a_series(series_table,dates,column_names): 

 #   series_dict_elem=series_dict_elem.astype(str)  
 
#    print("1series table shaper",series_table,series_table.shape)
#    series_table = series_table.reindex(natsorted(series_table.columns), axis=1)
 #   print("2series_tsable=",series_table.shape)
 #   series_table = series_table.reindex(natsorted(series_table.columns), axis=1)
  #  series_table=series_table.T 
 #   print("series_table.columns",series_table.columns)
   # # dates=pd.to_timestamp(series_table.index,freq="d",how="S").to_list()
  #  ndates=series_table.index.astype(str).tolist()
  #  print("dates=",len(dates))

    series_table=series_table.T
  #  print("2series_tsable=",series_table.shape)
 
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
    
   # print("series table before sorting=\n",series_table,series_table.shape)
    series_table = series_table.reindex(natsorted(series_table.columns), axis=1)
   # print("series table after sorting=\n",series_table,series_table.shape)

    
 #   print("\ngraph a series, table.T=\n",series_table,series_table.shape)
 #   print("3series_table.shape",series_table.shape,len(dates))
#    series_table=np.num_to_nan(series_table,0)
 #   series_table[column_names] = series_table[column_names].replace({0:np.nan})
 #   print("graph a series - series_table",series_table)
  #  series_table=series_table.T 
  #  print("4series_table.shape=\n",series_table,series_table.shape,len(dates))
#
    ax = plt.gca()

  #  del cols[-1]  # delete reference to period column
#    print("cols=",cols,len(cols))
    col_count=0
  #  print("Column names",column_names)
    for col in list(series_table.columns):
        if col=='period':
            pass
        else:    
      #      print("series_table,[col]col=",col)
            series_table[col] = series_table[col].replace({0:np.nan})
      #      print("\ngraph a series - series_table",col,"\n",series_table[col])
      
          #  print("find series type",col,"=",find_series_type(col))  
            series_suffix= str(find_series_type(col)) 
     #       print("series suffix=",series_suffix)
          #  series_type=str(series_dict[series_suffix])   # name, type of plot, colour
       #     print("series type=\n",series_type,">",series_type)   # name, type of plot, colour
            if (series_suffix=="mt_pred_mc"): # | (series_suffix=="mt_yerr_mc")):
                pred_plot=col
          #      print("pred_polt=",pred_plot)
    
            if col in column_names:
         #       print("add a series",col,series_suffix)
    #            series_table.plot(kind=series_dict_elem[1],x='period',y=col,color=series_dict_elem[2],ax=ax,fontsize=8)
                 #    plt.errorbar('period', series_table[col], yerr=series_table.iloc[col_count+1], data=series_table)
                if series_suffix=="mt_yerr_mc":
           #         print("\nplotting error bar\n")
                    plt.errorbar('period', pred_plot, yerr=col, fmt="r.",ms=3,data=series_table,ecolor="magenta",errorevery=1)
                   # plt.errorbar(series_table['period'], pred_plot, yerr=col, fmt="k.",ms=2,data=series_table,ecolor="magenta",errorevery=2)
               #     plt.errorbar(series_table.iloc[start_point:, series_table.columns.get_loc('period')], pred_plot, yerr=col, fmt="k.",ms=2,data=series_table,ecolor="magenta",errorevery=2)

      
                else:   
                    if series_suffix=="mt_":
                         plt.plot(series_table['period'],series_table[col],"b-",markersize=3,label=col)    #,range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
                    #     plt.plot(series_table['period'],series_table[col],"b-",markersize=3,label=col)    #,range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
    
                    elif series_suffix=="mt_pred_mc":        
                   #     plt.plot(series_table['period'],series_table[col],"g.",markersize=3,label=col) 
                        plt.plot(series_table['period'],series_table[col],"g.",markersize=3,label=col) 
    
                    else: 
                      #  pass
                        plt.plot(series_table['period'],series_table[col],"k.",markersize=3,label=col) 
                 #   series_table.plot(kind='scatter',x='period',y=col,color=series_type,ax=ax,fontsize=8,s=2,legend=False)
                  #      series_table.plot(kind='line',x='period',y=col,color=series_type,ax=ax,fontsize=8)
    
        col_count+=1    
        
    return 
    
   
    
    
    
def main():
    
    predict_ahead_steps=c.predict_ahead_steps  #130
    
     #   epochs_cnn=1
    epochs_wavenet=c.epochs_wavenet   #4
    no_of_batches=c.no_of_batches   #10000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=c.batch_length  #16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
    #    y_length=1
    neurons=c.neurons  #1600  #1000-2000
     
    #pred_error_sample_size=40
    
    patience=c.patience #6   #5
    
    # dictionary mat type code :   aggsum field, name, color
       # mat_type_dict=dict({"u":["qty","units","b-"]
                      #  "d":["salesval","dollars","r-"],
                      #  "m":["margin","margin","m."]
    #                   })
       
    mats=c.mats #[14]   #omving average window periods for each data column to add to series table
    start_point=c.start_point   #np.max(mats)+15  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
    mat_types=c.mat_types  #["u"]  #,"d","m"]
       
    # units_per_ctn=8
       
    # # train validate test split 
    # train_percent=0.7
    # validate_percent=0.2
    # test_percent=0.1
    
      
    
    # print("moving average days",mats)
    # print("start point",start_point)
    # print("predict ahead steps=",predict_ahead_steps,"\n")
    # required_starting_length=731+np.max(mats)+batch_length   # 2 years plus the MAT data lost at the start + batchlength
    
    
    
    #print(batch_dict)
    print("\n\nLearn from the batches")  
    print("======================\n")
    with open("batch_dict.pkl", "rb") as f:
        batches = pickle.load(f)
      #  testout2 = pickle.load(f)
    qnames=[batches[k][0] for k in batches.keys()]    
    print("unpickled",len(batches),"tables (",qnames,")")
    
    #for n in range(len(qnames)):    
    #    print(batches[n][0],"=\n",batches[n][1:7]) 
    
    
    
    
    
       
    ########################################
    model_list = defaultdict(list)
    model_filename_list=[]
    
    query_number=0
    for b in batches.keys():
        
         queryname=batches[b][0]   
         X_train=batches[b][1] 
         y_train =batches[b][2]
         X_valid=batches[b][3]
         y_valid =batches[b][4]
         X_test=batches[b][5]
         y_test =batches[b][6]
         mat_sales_x=batches[b][7]
         product_names=batches[b][8]
         series_table=batches[b][9]
        
         print("\n processing query:",queryname,"....\n")
         
         
     
       #  print("loading product_names")  #,product_names)
       #  with open('product_names.pkl', 'rb') as f:
       #       product_names = pickle.load(f)   
       #  print("product names=",product_names)     
    #    product_names=list(np.load("product_names.npy"))
       # # dates=list(np.load("periods.npy",allow_pickle=True))
         print("loading dates")
        # with open('dates.pkl', 'rb') as f:
        #     dates = pickle.load(f)   
         with open('extended_dates.pkl', 'rb') as f:
              extended_dates = pickle.load(f)   
       #     print("len dates",len(dates))
       #     print("\nlen extended dates",len(extended_dates),"\n")
         
         series_table= pd.read_pickle("series_table.pkl")
         print("Loading pivot table",series_table.shape) 
    
    
      #   print("Loading mat_sales_x")
      #   mat_sales_x=np.load("mat_sales_x.npy")
        
         actual_days_in_series_table=mat_sales_x.shape[1]
        
               
    #        product_names=list(series_table.index) 
    #        print("\nProduct names, length=",product_names,len(product_names))
    
         periods_len=actual_days_in_series_table
     #       print("PERIODS=",periods_len)
    
         n_query_rows=X_train.shape[2]
         n_steps=X_train.shape[1]-1
         n_inputs=X_train.shape[2]
         max_y=np.max(mat_sales_x)
          
         original_product_names=product_names
            
    
       # print("epochs_cnn=",epochs_cnn)
         print("epochs_wavenet=",epochs_wavenet)
       # print("dates=",dates)
        
      #   print("n_query_rows=",n_query_rows)    
         print("batch_length=",batch_length)
         print("n_inputs=",n_inputs)
         print("predict_ahead_steps=",predict_ahead_steps)
         print("full prediction day length=",periods_len)
    
         print("max y=",max_y)
    
    
    
     #   print("mini_batches X shape=",X[0],X.shape)  
     #   print("mini_batches y shape=",y[0],y.shape)  
       
      #  batch_size=X.shape[0]
      #  print("Batch size=",batch_size)
    
    
      # print("\npredict series shape",series.shape)
         print("X_train shape, y_train",X_train.shape, y_train.shape)
         print("X_valid shape, y_valid",X_valid.shape, y_valid.shape)
         print("X_test shape, y_test",X_test.shape, y_test.shape)
       
     #########################################################
        
      #  answer=input("Retrain model(s)?")
      #  if answer=="y":
            
         print("\n Neurons=",neurons,"[wavenet]. Building and compiling model\n")
        
         np.random.seed(42)
         tf.random.set_seed(42)
         layer_count=1    
    
         model = keras.models.Sequential()
         model.add(keras.layers.InputLayer(input_shape=[None,n_query_rows]))
        #if (n_inputs>=8):
         model.add(keras.layers.AlphaDropout(rate=0.2))
         model.add(keras.layers.BatchNormalization())
         for rate in (1,2,4,8) *2:      
            model.add(keras.layers.Conv1D(filters=neurons, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate)) 
            layer_count+=1    
         model.add(keras.layers.Conv1D(filters=n_query_rows, kernel_size=1))    
     #   optimizer=keras.optimizers.adam(lr=0.01,decay=1e-4)    
       # model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
         model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
       
        
       #     tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    #    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),MyCustomCallback()]
    #        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience),MyCustomCallback(),tensorboard_cb]
         callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),MyCustomCallback()]
    
    #    callbacks=[] 
        
         model.summary()
    
        
        # This callback will stop the training when there is no improvement in
        # the validation loss for three consecutive epochs.
    
                   #   ,tf.keras.callbacks.ModelCheckpoint(
                   #       filepath='mymodel_{epoch}',
                   # # Path where to save the model
                   # # The two parameters below mean that we will overwrite
                   # # the current checkpoint if and only if
                   # # the `val_loss` score has improved.
                   #   save_best_only=True,
                   #   monitor='val_loss',
                   #   verbose=1)
                   #  ]
    
       #     tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
         history = model.fit(X_train, y_train, epochs=epochs_wavenet, callbacks=callbacks,
                            validation_data=(X_valid, y_valid))
                
            
    
         model_filename="SCBS_model_"+str(qnames[query_number])+".h5"
         print("\nsave model '",model_filename,"'")
         model.save(model_filename, include_optimizer=True)
      
         model_filename_list.append(model_filename)   
      
       #  model_list[query_number].append(model)   
    #    model.summary()
       #     model.evaluate(X_valid, Y_valid)
    
    
        # Evaluate the model on the test data using `evaluate`
         print('\n# Evaluate on test data')
         results = model.evaluate(X_test, y_test, batch_size=5)
         print('test loss, test acc:', results)
    
          #  print('\nhistory dict:', history.history)
    
         #   print("plot learning curve")
         #   plot_learning_curves("learning curve",epochs_wavenet,history.history["loss"], history.history["val_loss"])
         #   plt.show()
     
         print("plot log learning curve")       
         plot_log_learning_curves("Log learning curve - "+str(qnames[query_number]),epochs_wavenet,history.history["loss"], history.history["val_loss"],qnames[query_number])
         plt.show()
    
    
    
     
         query_number+=1
         #########################################################
     
    print("final model filename list",model_filename_list)  
    
      
    #model_dict = dict((k, tuple(v)) for k, v in model_list.items())  #.iteritems())
    
    
    #print("\n model dict=\n",model_dict)
    
    with open("model_filenames.pkl","wb") as f:
        pickle.dump(model_filename_list, f,protocol=-1)
        
    #querynames=[table_dict[k][0] for k in table_dict.keys()]    
    #print("pickling",len(model_dict))   #," (",[table_dict[k][0] for k in table_dict.keys()],")")
    
    
    #######################################################
    
    
    print("\n\ntest unpickling")  
    with open("model_filenames.pkl", "rb") as f:
         testout1 = pickle.load(f)
    #   #  testout2 = pickle.load(f)
    # qnames=[testout1[k][0] for k in testout1.keys()]    
    print("unpickled model filename list",testout1)
    
    # #query_dict2=testout1['query_dict']
    # #print("table dict two unpickled=",testout1)
    
    # #df2=testout1.keys()
    # #print(testout1.keys())
    # #print(testout1.values())
    # #print(testout1[1])  
    # print(testout1[0][1]) 
    # print(testout1[1][1])
    
    return


if __name__ == '__main__':
    main()

   
