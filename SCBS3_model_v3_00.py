#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:48:08 2020

@author: tonedogga
"""


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# try:
#     # %tensorflow_version only exists in Colab.
#  #   %tensorflow_version 2.x
#     IS_COLAB = True
# except Exception:
#     IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)




from tensorflow import keras
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


# Common imports
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
#import random
import datetime as dt
from collections import defaultdict
import gc

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


visible_devices = tf.config.get_visible_devices('GPU') 
print("tf.config.get_visible_devices('GPU'):",visible_devices)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors



# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
    
 
  
    
def load_series(no_of_batches,batch_length,start_point,end_point):    
    with open("batch_dict.pkl", "rb") as f:
         seriesbatches = pickle.load(f)
    mat_sales_x =seriesbatches[0][7]
    X=seriesbatches[0][10]
    print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))
    mat_sales_x=mat_sales_x.astype(np.int32)
    print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))

    print("loaded mat_sales x shape",mat_sales_x.shape)
    print("start point=",start_point)
    print("end_point=",end_point)
    
    mat_sales_trunc=mat_sales_x[:,start_point:end_point+1,:]
    print("trimmed mat_sales x shape",mat_sales_trunc.shape)
    print("batch len=",batch_length)
    
    print("X size=",X.nbytes,type(X))
    X=X.astype(np.int32)
    print("after X size=",X.nbytes,type(X))


    
    #at_steps=mat_sales_x.shape[1]
    #f mat_steps < start_point:
    #   print("err mat_steps",mat_steps,"<",start+point)
  #  mat_sales_x1=np.roll(mat_sales_x,100,axis=1)
  #  print("mat_sales_x1",mat_sales_x1.shape)
  #  mat_sales_x=np.stack([mat_sales_x,mat_sales_x1],axis=2)[...,0]
    # shape is [1,steps,inputs]
    # add 100 empty to time series
   # mat_sales_x=np.stack([np.zeros([1])])
   
    #series_batch=build_mini_batches(mat_sales_x,no_of_batches,total_steps)   #,mat_sales_x.shape[2]) 
    #print("series_batch.shape",series_batch.shape)

    # extend the series by 365 days   
 #   extend=np.zeros(shape=[1,blank_future_days,1])
    #print("extend shape",extend.shape)
 #   mat_sales_x1=np.concatenate([mat_sales_x,extend],axis=1)
 #   print("new mat sales x shape",mat_sales_x1.shape)
    
#    X=build_mini_batches(mat_sales_x,no_of_batches,batch_length,start_point,end_point)  #,mat_sales_x.shape[2]) 
#    print("X.shape",X.shape)
  #  fullbatches=build_mini_batches(mat_sales_x,no_of_batches,mat_sales_x.shape[1])  #,mat_sales_x.shape[2]) 
  #  print("fullbatches.shape",fullbatches.shape)

    
    #series_table=batches[0][9]    
    return X,mat_sales_trunc,X.shape[0],mat_sales_x   #[..., np.newaxis].astype(np.float32)    
  

  
def create_batches(no_of_batches,batch_length,mat_sales_x,start_point,end_point):    
 #   print("mat_sales_x=\n",mat_sales_x[0])
 #   print("nob=",no_of_batches)
    if no_of_batches==1:
     #   print("\nonly one \n")
        repeats_needed=1
#        gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,end_point-start_point-start_point-batch_length+1))
     #  gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))

        gridtest=np.meshgrid(np.arange(0,batch_length),np.random.randint(0,end_point-start_point-batch_length+1))
   #     print("raandom",gridtest)
    else:    
        repeats_needed=int(no_of_batches/(end_point-batch_length-start_point)+1)  #      repeats_needed=int(no_of_batches/(end_point-start_point-start_point-batch_length))

        gridtest=np.meshgrid(np.arange(0,batch_length),np.arange(0,end_point-start_point-batch_length+1))  #int((end_point-start_point)/batch_length)+1))
 #   print("gi=\n",gridtest)
    start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
 #   print("start index=",start_index,start_index.shape)
    np.random.shuffle(start_index)
#    print("start index min/max=",np.min(start_index),np.max(start_index),start_index.shape) 

    X=mat_sales_x[0,start_index,:]
    np.random.shuffle(X)
 #   print("X.shape=\n",X.shape)
    gc.collect()
    return X   #,new_batches[:,1:batch_length+1,:]


    
# def load_series2(no_of_batches, n_steps):    
#     with open("batch_dict.pkl", "rb") as f:
#          batches = pickle.load(f)
#     mat_sales_x =batches[0][7]
#     print("loaded mat_sales x shape",mat_sales_x.shape)
    
#     # shape is [1,steps,inputs]
#     # add 100 empty to time series
#     #mat_sales_x=np.stack([np.zeros([1])])
    
    
#     #series_table=batches[0][9]    
#     series_batch=build_mini_batches(mat_sales_x,no_of_batches,n_steps) 
#     print("series_batch.shape",series_batch.shape)
#     return series_batch   #[..., np.newaxis].astype(np.float32)    
    

  
#  #mat_sales_pred,no_of_batches,batch_length,start_point,end_point
# #mat_sales_pred,no_of_batches,batch_length,start_point,end_point)#
# def build_mini_batches(mat_sales_orig,no_of_batches,batch_length,start_point,end_point):
# #    print("build",no_of_batches,"mini batches")
#   #  batch_length+=1  # add an extra step which is the target (y)
# #    np.random.seed(45)
#  #   print("bmb mat_sales_x.shape",mat_sales_x.shape)  
#     #total_steps=batch_length-start_point
#    # print("total steps=",total_steps)
#  #   print("series.shape",series.shape,"n_steps=",n_steps)
#  #   print("start_point=",start_point)
#  #   print("end point",end_point)
#  #   print("batch len=",batch_length)
#  #   print("mat_sales_pred",mat_sales_pred.shape)
#  #   print("total stesp=",total_steps)
#   #  print("no_of_batches to build=",no_of_batches)
#  #   print("no of steps in each batch=",n_steps)
#     if batch_length>(end_point-start_point):
#      #   print("\nonly one \n")
#         repeats_needed=1
# #        gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,end_point-start_point-start_point-batch_length+1))
#      #  gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))

#         gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))
#    #     print("raandom",gridtest)
#     else:    
#         repeats_needed=no_of_batches/int(((end_point-start_point)/batch_length)+1)  #      repeats_needed=int(no_of_batches/(end_point-start_point-start_point-batch_length))

#         gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.arange(0,int(((end_point-start_point)/batch_length)+1)))   #int((end_point-start_point)/batch_length)+1))
#   #  print("repeats needed=",repeats_needed)
#     #gridtest=np.meshgrid(np.arange(np.random.random_integers(0,total_steps,n_steps))), np.arange(0,n_steps))
#   #  print(gridtest,len(gridtest) ) #.shape)
#     start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
#   #  print("start index.shape",start_index,start_index.shape)
#     np.random.shuffle(start_index)
#    # print("start index",start_index,start_index.shape)
 
#     new_batches=mat_sales_orig[0,start_index]
#     np.random.shuffle(new_batches)
#   #  print("new batches",new_batches)
#     #if repeats_needed==1:
#       #  print(" one only - batches complete. batches shape:",new_batches.shape)
    
#     return new_batches   #,new_batches[:,1:batch_length+1,:]


# def create_Y(X,pred_length):   #,start_point,end_point):
#    # print("Y total_existing_steps",total_existing_steps)
#     batch_length=X.shape[1]
#     print("X batch length=",batch_length)
#   #  print("start point=",start_point)
#   #  print("end point",end_point)
#   #  pred_length=end_point-start_point
#     print("pred length",pred_length)
#  #   Y_window_length=(end_point-start_point)-pred_length
#  #   print("Y window length=",Y_window_length)
#     Y = np.empty((X.shape[0], batch_length-pred_length, pred_length),dtype=np.int32)
#   #  Y = np.empty((X.shape[0], batch_length, pred_length))
 
#     print("new Y shape",Y.shape)
#     for step_ahead in range(1, pred_length + 1):
#         Y[...,step_ahead - 1] = X[..., step_ahead:step_ahead+batch_length-pred_length, 0]  #+1
#    #     Y[...,step_ahead - 1] = X[..., step_ahead:step_ahead+batch_length+1, 0]  #+1

#     print("final create Y.shape=",Y.shape)

#     return Y




def plot_series(series, y=None, y_pred=None, x_label="$date$", y_label="$units/day$"):
    plt.plot(series, "-")
    if y is not None:
        plt.plot(n_steps, y, "b-")   #, markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "r-")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=90)
    plt.hlines(0, 0,  series.shape[0], linewidth=1)
  #  plt.axis([0, n_steps + 1, -1, 1])
    plt.axis([0, series.shape[0], 0 , np.max(series)])


# def plot_multiple_forecasts(X, Y, Y_pred,title_name):
#     n_steps = X.shape[1]
#     ahead = Y.shape[1]
#     plot_series(X[0, :, 0])
#     if title_name:
#         plt.title(title_name)
#     plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "r-", label="Actual")
#     plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "b-", label="Forecast", markersize=10)
# #    plt.axis([0, n_steps + ahead, -1, 1])
#     plt.axis([0, n_steps + ahead, 0 , np.max([np.max(Y),np.max(Y_pred)])])

#     plt.legend(fontsize=14)

#mat_sales_pred,mat_sales_x,title+" days:"+str(days),first_start_point,first_end_point
def plot_multiple_forecasts2(X_orig,X_pred, X_actual,title_name,first_start_point,first_end_point):
    n_steps = X_pred.shape[1]
    ahead = X_actual.shape[1]
    n_orig=X_orig.shape[1]-first_start_point
 #   plt.plot(np.arange(0, n_steps), X[0, :, 0], "r-", label="forecast", markersize=10)
 
  #  plt.plot(np.arange(0, ahead), X_actual[0, :, 0], "g-", label="Actual", markersize=10)


    plt.plot(np.arange(0, n_orig), X_orig[0, first_start_point:, 0], "b-", label="Actual", markersize=10)

 #   plot_series(X[0, :, 0])
    if title_name:
        plt.title(title_name)
    plt.plot(np.arange(0,n_steps), X_pred[0, :, 0], "r.", label="Predict",markersize=4)
#    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "b-", label="Forecast", markersize=10)
 
#    plt.axis([0, n_steps + ahead, -1, 1])
#    plt.axis([0, n_steps + ahead, 0 , np.max(Y_actual)])
    plt.axis([0, n_steps+10, 0 , np.max(X_actual)])

    plt.legend(fontsize=14)





def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def save_fig(fig_id, images_path, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
 
    
 
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
    
   
    
    
    
def main(c):
  #  images_path = os.path.join(c.output_dir, "images/")
    print("\noutput dir=",c.output_dir)
    print("images path=",c.images_path)
#
#IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
  #  os.makedirs(images_path, exist_ok=True)

    predict_ahead_length=c.predict_ahead_length  #130
    
     #   epochs_cnn=1
    epochs=c.epochs   #4
    no_of_batches=c.no_of_batches   #10000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=c.batch_length  #16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
    #    y_length=1
    neurons=c.neurons  #1600  #1000-2000
    end_point=c.end_point
    start_point=c.start_point
    
    X_window_length =c.X_window_length   #=c.batch_length-c.predict_ahead_length

    #pred_error_sample_size=40
    
    patience=c.patience #6   #5
    
    # dictionary mat type code :   aggsum field, name, color
       # mat_type_dict=dict({"u":["qty","units","b-"]
                      #  "d":["salesval","dollars","r-"],
                      #  "m":["margin","margin","m."]
    #                   })
       
    mats=c.mats #[14]   #omving average window periods for each data column to add to series table
   # start_point=c.start_point   #np.max(mats)+15  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
    mat_types=c.mat_types  #["u"]  #,"d","m"]
    units_per_ctn=c.units_per_ctn  #8
 
    dates=c.dates    
 
    dropout_rate=c.dropout_rate
    # units_per_ctn=8
       
    # # train validate test split 
    # train_percent=0.7
    # validate_percent=0.2
    # test_percent=0.1
    
    train_percent= c.train_percent
    validate_percent=c.validate_percent
    test_percent=c.test_percent
 
    images_path=c.images_path
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
         
   
       
    # # train validate test split 
    # train_percent=c.train_percent   #0.7
    # validate_percent=c.validate_percent  #0.2
    # test_percent=c.test_percent  #0.1
     
    
    # train_size=int(round((no_of_batches*train_percent),0))
    # validate_size=int(round((no_of_batches*validate_percent),0))
    # test_size=int(round((no_of_batches*test_percent),0))


    print("moving average days",mats)
    print("start point",start_point)
    print("predict ahead length=",predict_ahead_length,"\n")
    required_starting_length=c.required_starting_length    #=731+np.max(mats)+batch_length   # 2 years plus the MAT data lost at the start + batchlength
    
    
    print("\nBatch creator\n\n")
    print("unpickling '","tables_dict.pkl","'")  
    with open("tables_dict.pkl", "rb") as f:
        all_tables = pickle.load(f)
      #  testout2 = pickle.load(f)
    qnames=[all_tables[k][0] for k in all_tables.keys()]    
    print("unpickled",len(all_tables),"tables (",qnames,")")
    
    
    
    
    #n_steps = 50
  #  shortened_series,mat_sales_x,dates = load_series(start_point,end_point)
  #  print("shoerened series.shape=",shortened_series.shape)
    print("len dates=",len(dates))
    #print("mat_sales_x [:,2:]=",mat_sales_x[:,1:].shape)
    #print("mat_sales_x[:,:-1]=",mat_sales_x[:,:-1].shape)
    
          
        
        
       
    ########################################
    model_list = defaultdict(list)
    model_filename_list=[]
    
    query_number=0
    for b in batches.keys():
        
         queryname=batches[b][0]   
         # X_train=batches[b][1] 
         # Y_train =batches[b][2]
         # X_valid=batches[b][3]
         # Y_valid =batches[b][4]
         # X_test=batches[b][5]
         # Y_test =batches[b][6]
         mat_sales_x=batches[b][1]
         product_names=batches[b][2]
         series_table=batches[b][3]
  #       X=batches[b][10]
        
         print("\n processing query:",queryname,"....\n")
         mat_sales_orig=mat_sales_x
         mat_sales_pred=mat_sales_x
        # #n_steps = 50
        
            
         print("series table shape=",series_table.shape)     
         X=create_batches(no_of_batches,batch_length,mat_sales_x[:,:-1],start_point,end_point)
        #print("X.shape",X[0],X.shape)
         print("X size=",X.nbytes,"bytes")
        
         n_query_rows=X.shape[2]
         n_steps=X.shape[1]-1
         n_inputs=X.shape[2]
         max_y=np.max(X)
 
    
         print("n_query_rows=",n_query_rows)    
         print("batch_length=",batch_length)
         print("n_inputs=",n_inputs)
         print("predict_ahead_length=",predict_ahead_length)
    #     print("full prediction day length=",periods_len)
        
         print("max y=",max_y)
        
    
            
         
         n_train=int(round((no_of_batches*train_percent),0))
         n_validate=int(round((no_of_batches*validate_percent),0))
         n_test=int(round((no_of_batches*test_percent),0))
         
        
            
 
    
 
         X_train = X[:n_train]
         X_valid = X[n_train:n_train+n_validate]
         X_test = X[n_train+n_validate:]
        
        
        
        #Y=create_batches(no_of_batches,batch_length,mat_sales_x[:,1:],start_point,end_point)
        #print("start_Y.shape",start_Y[0],start_Y.shape)#
        
         Y = np.empty((no_of_batches, batch_length,predict_ahead_length))
         
         print("new Y shape",Y.shape)
         for step_ahead in range(1, predict_ahead_length + 1):
        #    Y[:,:,step_ahead - 1] = shortened_series[:, step_ahead:step_ahead+batch_length,0]  #,n_inputs-1]  #+1
             Y[:,:,step_ahead - 1] = mat_sales_x[:, step_ahead:step_ahead+batch_length,0]  #,n_inputs-1]  #+1
         
        #   #      print("step a=",step_ahead,"X=",X[..., step_ahead:step_ahead+batch_length,0],"ss=",shortened_series[..., step_ahead:step_ahead+batch_length+1, 0])  #+1
        
        
        #Y=create_Y(shortened_series,X,sample_length,predict_ahead_steps)    #,start_point,end_point)
         print("Y.shape",Y.shape)
         print("Y size=",Y.nbytes,"bytes")
         Y_train = Y[:n_train]
         Y_valid = Y[n_train:n_train+n_validate]
         Y_test = Y[n_train+n_validate:]
        
         no_of_batches=X.shape[0]
         batch_lemgth=X.shape[1]
         n_inputs=X.shape[2]
        
         print("start point",start_point)
         print("end point",end_point)
        
         print("no of batches=",no_of_batches)
         print("batch length",batch_length)
         print("n_inputs",n_inputs)
    #     print("sample length",sample_length)
        
         print("X_train",X_train.shape)
         print("Y_train",Y_train.shape)
         print("X_valid",X_valid.shape)
         print("Y_valid",Y_valid.shape)
         print("X_test",X_test.shape)
         print("Y_test",Y_test.shape)
        
 
        
        
        
        
 #           mat_sales_x =seriesbatches[0][7]
         print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))
         mat_sales_x=mat_sales_x.astype(np.int32)
         print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))

         print("loaded mat_sales x shape",mat_sales_x.shape)
         print("start point=",start_point)
         print("end_point=",end_point)
    
         mat_sales_trunc=mat_sales_x[:,start_point:end_point+1,:]
         print("trimmed mat_sales x shape",mat_sales_trunc.shape)
         print("batch len=",batch_length)
 
        # X,mat_sales_x,no_of_batches,mat_sales_orig = load_series(no_of_batches,batch_length,start_point,end_point)

 
     
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
         print("epochs=",epochs)
       # print("dates=",dates)
        
      #   print("n_query_rows=",n_query_rows)    
         print("batch_length=",batch_length)
         print("n_inputs=",n_inputs)
         print("predict_ahead_length=",predict_ahead_length)
         print("full prediction day length=",periods_len)
    
         print("max y=",max_y)
 
##############################    


         n_train=int(round((no_of_batches*train_percent),0))
         n_validate=int(round((no_of_batches*validate_percent),0))
         n_test=int(round((no_of_batches*test_percent),0))

#train_size=int(round(batch_size*train_percent,0))
        

 
     #     Y=create_Y(X,predict_ahead_length)    #,start_point,end_point)
     #     #print("Y size=",Y.nbytes,type(Y))
     #     #Y=Y.astype(np.int32)
     #     print("Y size=",Y.nbytes,type(Y))
        
     #    # Y = np.empty((batched_series.shape[0], n_steps, predict_ahead))
     #    # for step_ahead in range(1, predict_ahead + 1):
     #    #     Y[..., step_ahead - 1] = batched_series[..., step_ahead:step_ahead + n_steps, 0]
     #     print("create Y.shape=",Y.shape)
     #     Y_train = Y[:n_train]
     #     Y_valid = Y[n_train:n_train+n_validate]
     #     Y_test = Y[n_train+n_validate:]
        
        
     #     print("Y shape",Y.shape)
        
     #     print("X_train",X_train.shape)
     #     print("Y_train",Y_train.shape)
     #     print("X_valid",X_valid.shape)
     #     print("Y_valid",Y_valid.shape)
     #     print("X_test",X_test.shape)
     #     print("Y_test",Y_test.shape)
        

    
    
     # #   print("mini_batches X shape=",X[0],X.shape)  
     #   print("mini_batches y shape=",y[0],y.shape)  
       
      #  batch_size=X.shape[0]
      #  print("Batch size=",batch_size)
    
    
      # print("\npredict series shape",series.shape)
         print("X_train shape, Y_train",X_train.shape, Y_train.shape)
         print("X_valid shape, Y_valid",X_valid.shape, Y_valid.shape)
         print("X_test shape, Y_test",X_test.shape, Y_test.shape)
       
     #########################################################
        
      #  answer=input("Retrain model(s)?")
      #  if answer=="y":
            
         print("\n Neurons=",neurons,". Building and compiling model\n")
        
         print("GRU with batch norm and dropout")
        
                ##############################################################
  #       gc.collect()
# =============================================================================
#          answer=input("load saved model?")
#          if answer=="y":
#             print("\n\nloading model...")  
#             model=keras.models.load_model("GRU_Dropout_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
#          else:
# 
#  Now let's create an RNN that predicts the next 10 steps at each time step. 
# That is, instead of just forecasting time steps 50 to 59 based on time steps 0 to 49,
#  it will forecast time steps 1 to 10 at time step 0, then time steps 2 to 11 at time 
# step 1, and so on, and finally it will forecast time steps 50 to 59 at the last time step.
#  Notice that the model is causal: when it makes predictions at any time step, 
# it can only see past time steps.

#print("cretae an RNN that predicts the next 10 steps at each time step")

            
# =============================================================================
        
       #  print("GRU with dropout")
        
         np.random.seed(42)
         tf.random.set_seed(42)
        
         model = keras.models.Sequential([
            keras.layers.GRU(neurons, return_sequences=True, input_shape=[None, n_inputs]),
      #      keras.layers.LSTM(neurons, return_sequences=True, input_shape=[None, n_inputs]),

            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.BatchNormalization(),
            keras.layers.GRU(neurons, return_sequences=True),
      #      keras.layers.LSTM(neurons, return_sequences=True),

            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.BatchNormalization(),
            keras.layers.TimeDistributed(keras.layers.Dense(predict_ahead_length))
         ])
        
        
         opt = keras.optimizers.Adam(learning_rate=0.1)    #0.03
         model.compile(loss="mse", optimizer=opt, metrics=[last_time_step_mse])

   #      model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        
         callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),MyCustomCallback()]
        
         history = model.fit(X_train, Y_train, epochs=epochs,
                            validation_data=(X_valid, Y_valid))
        
        
         print("\nsave model\n")
         model.save("GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
           
         model.summary()
    
         plot_learning_curves(history.history["loss"], history.history["val_loss"],epochs,"GRU and dropout")
         plt.show()


###################################3
     
        
                
        
        
    #      np.random.seed(42)
    #      tf.random.set_seed(42)
    #      layer_count=1    
    
    #      model = keras.models.Sequential()
    #      model.add(keras.layers.InputLayer(input_shape=[None,n_query_rows]))
    #     #if (n_inputs>=8):
    #      model.add(keras.layers.AlphaDropout(rate=c.dropout_rate))
    #      model.add(keras.layers.BatchNormalization())
    #      for rate in (1,2,4,8) *2:      
    #         model.add(keras.layers.Conv1D(filters=neurons, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate)) 
    #         layer_count+=1    
    #      model.add(keras.layers.Conv1D(filters=n_query_rows, kernel_size=1))    
    #  #   optimizer=keras.optimizers.adam(lr=0.01,decay=1e-4)    
    #    # model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    #      model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
       
        
    #    #     tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    # #    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),MyCustomCallback()]
    # #        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience),MyCustomCallback(),tensorboard_cb]
    #      callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),MyCustomCallback()]
    
    # #    callbacks=[] 
        
    #      model.summary()
    
        
    #     # This callback will stop the training when there is no improvement in
    #     # the validation loss for three consecutive epochs.
    
    #                #   ,tf.keras.callbacks.ModelCheckpoint(
    #                #       filepath='mymodel_{epoch}',
    #                # # Path where to save the model
    #                # # The two parameters below mean that we will overwrite
    #                # # the current checkpoint if and only if
    #                # # the `val_loss` score has improved.
    #                #   save_best_only=True,
    #                #   monitor='val_loss',
    #                #   verbose=1)
    #                #  ]
    
    #    #     tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    #      history = model.fit(X_train, y_train, epochs=epochs_wavenet, callbacks=callbacks,
    #                         validation_data=(X_valid, y_valid))
                
            
    
         model_filename="SCBS_model_"+str(qnames[query_number])+".h5"
         print("\nsave model '",model_filename,"'")
         model.save(model_filename, include_optimizer=True)
      
         model_filename_list.append(model_filename)   
      
       #  model_list[query_number].append(model)   
    #    model.summary()
       #     model.evaluate(X_valid, Y_valid)
    
    
        # Evaluate the model on the test data using `evaluate`
    #     print('\n# Evaluate on test data')
    #     results = model.evaluate(X_test, Y_test, batch_size=5)
    #     print('test loss, test acc:', results)
    
          #  print('\nhistory dict:', history.history)
    
         #   print("plot learning curve")
         #   plot_learning_curves("learning curve",epochs_wavenet,history.history["loss"], history.history["val_loss"])
         #   plt.show()
     
     #    print("plot log learning curve")       
     #    plot_log_learning_curves("Log learning curve - "+str(qnames[query_number]),c.images_path,epochs_wavenet,history.history["loss"], history.history["val_loss"],qnames[query_number])
     #    plt.show()
    
    
    
     
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
    
    
    # print("\n\ntest unpickling")  
    # with open("model_filenames.pkl", "rb") as f:
    #      testout1 = pickle.load(f)
    # #   #  testout2 = pickle.load(f)
    # # qnames=[testout1[k][0] for k in testout1.keys()]    
    # print("unpickled model filename list",testout1)
    
    # # #query_dict2=testout1['query_dict']
    # # #print("table dict two unpickled=",testout1)
    
    # # #df2=testout1.keys()
    # #print(testout1.keys())
    # #print(testout1.values())
    # #print(testout1[1])  
    # print(testout1[0][1]) 
    # print(testout1[1][1])
    
    return


if __name__ == '__main__':
    main()



















# ###########################################3

# patience=10
# epochs=10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
# start_point=250
# no_of_batches=8000
# end_point=450
# # predict aherad length is inside batch_length
# predict_ahead_length=50
# batch_length=20+predict_ahead_length
# #batch_length=(end_point-start_point)+predict_ahead_length
# X_window_length=batch_length-predict_ahead_length
# #future_steps=400
# #blank_future_days=365
# # batch_total=100000
# train_percent=0.7
# validate_percent=0.2
# test_percent=0.1

# #  Now let's create an RNN that predicts the next 10 steps at each time step. 
# # That is, instead of just forecasting time steps 50 to 59 based on time steps 0 to 49,
# #  it will forecast time steps 1 to 10 at time step 0, then time steps 2 to 11 at time 
# # step 1, and so on, and finally it will forecast time steps 50 to 59 at the last time step.
# #  Notice that the model is causal: when it makes predictions at any time step, 
# # it can only see past time steps.

# #print("cretae an RNN that predicts the next 10 steps at each time step")

# np.random.seed(42)

# #n_steps = 50
# X,mat_sales_x,no_of_batches,mat_sales_orig = load_series(no_of_batches,batch_length,start_point,end_point)


# n_train=int(round((no_of_batches*train_percent),0))
# n_validate=int(round((no_of_batches*validate_percent),0))
# n_test=int(round((no_of_batches*test_percent),0))



# print("X.shape",X.shape)
# print("mat_sales_x.shape",mat_sales_x.shape)
# n_inputs=X.shape[2]
# #total_existing_steps=mat_sales_x.shape[1]
# #total_steps=total_existing_steps+future_steps
# #end_point=start_point+batch_length
# #print("future_steps",future_steps)
# #print("total xisting steps",total_existing_steps)
# #print("total_steps=",total_steps)
# print("X_window_length=",X_window_length)
# print("start point",start_point)
# print("batch length",batch_length)
# #print("end point",end_point)
# print("n_inputs",n_inputs)


# X_train = X[:n_train, :X_window_length]
# X_valid = X[n_train:n_train+n_validate, :X_window_length]
# X_test = X[n_train+n_validate:, :X_window_length]

# # create_Y(batches_series,start_point,end_point,batch_length)
# #batched_series=create_new(batched_series,n_steps,predict_ahead)
# Y=create_Y(X,predict_ahead_length)    #,start_point,end_point)
# #print("Y size=",Y.nbytes,type(Y))
# #Y=Y.astype(np.int32)
# print("Y size=",Y.nbytes,type(Y))

# # Y = np.empty((batched_series.shape[0], n_steps, predict_ahead))
# # for step_ahead in range(1, predict_ahead + 1):
# #     Y[..., step_ahead - 1] = batched_series[..., step_ahead:step_ahead + n_steps, 0]
# print("create Y.shape=",Y.shape)
# Y_train = Y[:n_train]
# Y_valid = Y[n_train:n_train+n_validate]
# Y_test = Y[n_train+n_validate:]


# print("Y shape",Y.shape)

# print("X_train",X_train.shape)
# print("Y_train",Y_train.shape)
# print("X_valid",X_valid.shape)
# print("Y_valid",Y_valid.shape)
# print("X_test",X_test.shape)
# print("Y_test",Y_test.shape)


# ##############################################################
# #######################################3
# n_steps=mat_sales_x.shape[1]
# plot_series(mat_sales_x[0,:,0],Y_valid[0,0,0])
# plt.show()
# mat_sales_pred=mat_sales_x
# #mat_sales_orig=mat_sales_x
# #plot_series(X_valid[0, :, 0], Y_valid[0, 0])
# #plot_series(X[0,:,0])

# #print("mar shape",mat_sales_x[0,:,0].shape)

# print("GRU with batch norm and dropout")


# np.random.seed(42)
# tf.random.set_seed(42)

# model = keras.models.Sequential([
#     keras.layers.GRU(400, return_sequences=True, input_shape=[None, n_inputs]),
# #    keras.layers.Dropout(rate=0.2),
#     keras.layers.BatchNormalization(),
#     keras.layers.GRU(400, return_sequences=True),
#  #   keras.layers.Dropout(rate=0.2),
#     keras.layers.BatchNormalization(),
#     keras.layers.TimeDistributed(keras.layers.Dense(predict_ahead_length))
# ])

# model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])

# callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),MyCustomCallback()]

# history = model.fit(X_train, Y_train, epochs=epochs,
#                     validation_data=(X_valid, Y_valid))


# print("\nsave model\n")
# model.save("GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
   
# model.summary()

# plot_learning_curves(history.history["loss"], history.history["val_loss"],"GRU batch norm and dropout")
# plt.show()


# predict_ahead("GRU with dropout predictions",model,mat_sales_orig,mat_sales_pred,mat_sales_x,X,batch_length,X_window_length,predict_ahead_length,start_point,end_point)

