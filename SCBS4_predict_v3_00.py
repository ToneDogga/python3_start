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
from datetime import datetime
#import SCBS0 as c

#filename="tables_dict.pkl"



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

print("\n\nUse pretrained models to make predictions - By Anthony Paech 30/4/20")
print("========================================================================\n")       

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
#PROJECT_ROOT_DIR = "."
#CHAPTER_ID = "rnn"

#n_steps = 50

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
    save_fig(c.output_dir+"log_learning_curve "+str(query_name))
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












def save_fig(fig_id, images_path,tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
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
    
     
  
def add_a_new_series(table,arr_names,arr,start_point,predict_ahead_steps,periods_len):
  
  #  print("ans input table shape",table,table.shape)
  #  print("add a new series first date=",table.index[0])
  #  print("ans arr_names",arr_names)
  #  print("ans arr[0]",arr[0].shape)
    first_date=(table.T.index[0]+timedelta(days=int(start_point+1))).strftime('%Y-%m-%d')
  #  print("ans first_date",first_date)
    pidx = pd.period_range(table.T.index[0]+timedelta(days=int(start_point+1)), periods=periods_len-1-start_point)   # 2000 days  
  #  print("befofre dates=pidx",pidx,periods_len)
    
    pad_before_arr = np.empty((1,start_point,arr.shape[2]))
    pad_before_arr[:] = np.NaN
  #  print("pad befor arr=\n",pad_before_arr.shape,"arr[0]=\n",arr[:,start_point:,:].shape)
    y_values= np.concatenate((pad_before_arr,arr[:,start_point:,:]),axis=1)
   # print("aaseries y_values.shapoe",y_values[0].shape)
  #  print("pidx=",len(pidx))
#    new_cols=pd.DataFrame(arr[0,start_point:,:],columns=arr_names,index=pidx[start_point:]).T
 #   new_cols=pd.DataFrame(y_values,columns=arr_names,index=pidx[start_point:]).T
    new_cols=pd.DataFrame(y_values[0],columns=arr_names,index=pidx).T

  #  print("ans input new cols",new_cols,new_cols.shape)
  #  print("Table=\n",table,table.shape)
 #   table=table.T 
    table2=pd.concat((table,new_cols),join='outer',axis=0)   
    new_product_names=list(table2.T.columns)
  #  print("ans output table2 shape",table2,table2.shape)
#    print("extended dates",list(series_table.index))
    extended_dates=list(table2.columns.to_timestamp(freq="D",how="S"))
  #  print("extended dates=",extended_dates)
    return table2,new_product_names,extended_dates
    
     



# def log_dir(prefix=""):
#     now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#     root_logdir = "tf_logs"
#     if prefix:
#         prefix += "-"
#     name = prefix + "run-" + now
#     return "{}/{}/".format(root_logdir, name)
 




def build_mini_batches(mat_sales_orig,no_of_batches,batch_length,start_point,end_point):
    if batch_length>(end_point-start_point):
     #   print("\nonly one \n")
        repeats_needed=1
#        gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,end_point-start_point-start_point-batch_length+1))
     #  gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))

        gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))
   #     print("raandom",gridtest)
    else:    
        repeats_needed=no_of_batches/int(((end_point-start_point)/batch_length)+1)  #      repeats_needed=int(no_of_batches/(end_point-start_point-start_point-batch_length))

        gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.arange(0,int(((end_point-start_point)/batch_length)+1)))   #int((end_point-start_point)/batch_length)+1))
  #  print("repeats needed=",repeats_needed)
    #gridtest=np.meshgrid(np.arange(np.random.random_integers(0,total_steps,n_steps))), np.arange(0,n_steps))
  #  print(gridtest,len(gridtest) ) #.shape)
    start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
  #  print("start index.shape",start_index,start_index.shape)
    np.random.shuffle(start_index)
   # print("start index",start_index,start_index.shape)
 
    new_batches=mat_sales_orig[0,start_index]
    np.random.shuffle(new_batches)
  #  print("new batches",new_batches)
    #if repeats_needed==1:
      #  print(" one only - batches complete. batches shape:",new_batches.shape)
    
    return new_batches   #,new_batches[:,1:batch_length+1,:]














    
    
def actual_days(series_table):
 #   print("ad=",series_table.index[0])
    first_date=series_table.index[0].to_timestamp(freq="D",how="S")
    last_date=series_table.index[-1].to_timestamp(freq="D",how="S")
    return (last_date - first_date).days +1    #.timedelta_series.dt.days    



       

def find_series_type(series_name):
    return series_name[series_name.find(':')+1:]




def predict_ahead(title,model,mat_sales_orig,mat_sales_pred,mat_sales_x,X,batch_length,X_window_length,predict_ahead_length,start_point,end_point):
    no_of_batches=X.shape[0]
    
    first_start_point=start_point
    first_end_point=end_point
    
    starting_batch_length=batch_length
    
    starting_mat_sales_orig=mat_sales_orig
    predict_count=first_start_point
 #   print("X.shape",X.shape)
 #   print("X_window_length",X_window_length)
    for days in range(X_window_length,X_window_length+predict_ahead_length+1):
     #   print("days=",days)
        #print("predict X_new",model.predict(X_new).shape)
        X_new, Y_new = X[:, :days, :], X[:,days:, :]
      #  print("new X_new.shape",X_new.shape)
      #  print("new Y_new.shape",Y_new.shape)
        pred=model.predict(X_new).astype(np.int32)
    
        Y_pred_build = pred[:,:,-1][..., np.newaxis]
        Y_pred_build=Y_pred_build[:,-1][..., np.newaxis]
        Y_pred_build=Y_pred_build[0,:][np.newaxis,...]
     #   print("Y_pred_build.shape",Y_pred_build.shape)
    #    plot_multiple_forecasts3(mat_sales_pred,"finish Deep RNN with batch norm and dropout, days:"+str(days))
    #    plt.show()
 


    
       # print("before X.shape",X.shape)
     #   print("X_new=",X_new.shape)
     #   print("Y_new=",Y_new.shape)
    
        #print(" before X.shape",X.shape)
      #  X_new=np.concatenate([X_new,Y_pred_build],axis=1)
     #   print("mat sales pred.shape before",mat_sales_pred.shape)
        mat_sales_pred=np.concatenate([mat_sales_pred,Y_pred_build],axis=1)
   #     plot_series(mat_sales_pred[0,:,0],X_new[0,0,0])
    #    plot_multiple_forecasts(X_new,mat_sales_pred,Y_pred_build,"final Deep RNN with batch norm and dropout, days:"+str(days))

        mat_sales_orig[:,predict_count]=Y_pred_build

        print("prediction step:",pred.shape[1],"/",batch_length,"=",Y_pred_build)

    #    plot_multiple_forecasts2(mat_sales_orig,mat_sales_pred,mat_sales_x,title+" days:"+str(days),first_start_point,first_end_point)
      #  plt.show()
   
      #  plt.show()
      #  print("mat sales pred.shape after",mat_sales_pred.shape)
       # batch_length+=1
        end_point+=1
        start_point+=1
       
        # replace actual in mat_sales_orig with mat_sales_pred
        
        # print("before mat_sales_orig.shape",mat_sales_orig.shape)  
        # print("slice:", mat_sales_orig[:,predict_count,:])
        # print("before mat_sales_pred.shape",mat_sales_pred.shape)        
        # print("before y pred build",Y_pred_build,Y_pred_build.shape)
        
  

        # print("after mat_sales_orig.shape",mat_sales_orig.shape)  
        # print("after mat_sales_pred.shape",mat_sales_pred.shape)        

        # #print("afgter X.shape",X.shape)
      #  print("after X_new.shape",X_new.shape)
      #  X=build_mini_batches(mat_sales_orig,no_of_batches,batch_length,start_point,end_point)  #,mat_sales_x.shape[2]) 
        X=build_mini_batches(mat_sales_orig,no_of_batches,batch_length,start_point,end_point)  #,mat_sales_x.shape[2]) 

        #  print("X.shape",X.shape)
    #    X=build_mini_batches(mat_sales_x,no_of_batches,batch_length)  #,mat_sales_x.shape[2]) 
    #    print("X.shape",X.shape)
        #print("afgter X.shape",X.shape)
        predict_count+=1
    
    plot_multiple_forecasts2(starting_mat_sales_orig,mat_sales_pred,mat_sales_x,title+" days:"+str(days),first_start_point,first_end_point)
    plt.show()     
#   Y_pred = model.predict(X_new)[:,-1][..., np.newaxis]
    plot_multiple_forecasts2(mat_sales_orig,mat_sales_pred,mat_sales_x,title+" days:"+str(days),first_start_point,first_end_point)
    plt.show()
    
 #   plot_series(mat_sales_x[0,:,0],Y_valid[0,0,0])
   # plt.show()
 #   plot_series(mat_sales_pred[0,:,0],Y_valid[0,0,0])
 #   plt.show()
    return






def main(c):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

    print("\noutput dir=",c.output_dir)
    print("images path=",c.images_path)

    
    predict_ahead_length=c.predict_ahead_length  #130
    X_window_length=c.X_window_length
    start_point=c.start_point
    end_point=c.end_point
     #   epochs_cnn=1
    #epochs_wavenet=1
    #no_of_batches=10000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=c.batch_length #16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
    #    y_length=1
    #neurons=1600  #1000-2000
     
    pred_error_sample_size=c.pred_error_sample_size  #40
    
    #patience=6   #5
    
    # dictionary mat type code :   aggsum field, name, color
       # mat_type_dict=dict({"u":["qty","units","b-"]
                      #  "d":["salesval","dollars","r-"],
                      #  "m":["margin","margin","m."]
    #                   })
       
    mats=c.mats   #[14]   #omving average window periods for each data column to add to series table
    start_point=c.start_point   #np.max(mats)+15  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
    mat_types=c.mat_types # ["u"]  #,"d","m"]
       
    
    
    
    #print(batch_dict)
    print("\n\nPredict from models")  
    
    
    
    print("\n\nunpickling model filenames")  
    with open("model_filenames.pkl", "rb") as f:
         model_filenames_list = pickle.load(f)
    #   #  testout2 = pickle.load(f)
    # qnames=[testout1[k][0] for k in testout1.keys()]    
    print("unpickled model filenames list",model_filenames_list)
    
    with open("batch_dict.pkl", "rb") as f:
        batches = pickle.load(f)
    
    with open("tables_dict.pkl", "rb") as f:
        all_tables = pickle.load(f)
      #  testout2 = pickle.load(f)
    qnames=[all_tables[k][0] for k in all_tables.keys()]    
    print("unpickled",len(all_tables),"tables (",qnames,")")
      
    
    required_starting_length=c.required_starting_length  #731+np.max(mats)+batch_length   # 2 years plus the MAT data lost at the start + batchlength
    
     
    
    ########################################
    
    model_number=0
    for model_name in model_filenames_list:
        model=keras.models.load_model(model_name,custom_objects={"last_time_step_mse": last_time_step_mse})
    
        print("\nmodel=",model_name,"loaded.\n\n")
        model.summary
       
        with open("product_names_"+str(qnames[model_number])+".pkl","rb") as g:
             original_product_names=pickle.load(g)
     
     
        
        print("\nSingle Series Predicting",predict_ahead_length,"steps ahead.")
     #   print("series=\n",list(series_table.index),"\n")
        print("Loading mat_sales_x")
      #  mat_sales_x=np.load("mat_sales_x.npy")
      #  with open("batch_dict.pkl", "rb") as f:
      #      batches = pickle.load(f)
        mat_sales_x =batches[model_number][7]
        X=batches[model_number][10]
        
        
        mat_sales_orig=mat_sales_x
        mat_sales_pred=mat_sales_x
        
        product_names=batches[model_number][8]
        #series_table=all_tables[model_number][1]
        series_table=batches[model_number][9]
    
        print("unpickled series table shape",series_table.shape)
    
        print("mat_sales_x shape",mat_sales_x.shape)
        print("series table shape",series_table.shape)
    
        actual_days_in_series_table=actual_days(series_table.T)
            
        #series_table=series_table.T    
        print("actual days in series table=",actual_days_in_series_table)
        print("required minimum starting days for 2 year series analysis:",required_starting_length)
    
    
        periods_len=actual_days_in_series_table  #+predict_ahead_steps # np.max(mats)
        print("total periods=",periods_len)
    
    
        original_steps=mat_sales_x.shape[1]
        
#####################################################################3        

# Now let's create an RNN that predicts the next 10 steps at each time step. That is, 
# instead of just forecasting time steps 50 to 59 based on time steps 0 to 49, 
# it will forecast time steps 1 to 10 at time step 0, then time steps 2 to 11 at time step 1, 
# and so on, and finally it will forecast time steps 50 to 59 at the last time step. 
# Notice that the model is causal: when it makes predictions at any time step, it can 
# only see past time steps.    

              
        predict_ahead("GRU with dropout predictions",model,mat_sales_orig,mat_sales_pred,mat_sales_x,X,batch_length,X_window_length,predict_ahead_length,start_point,end_point)
  
      #  n_steps = 50
      #  series = generate_time_series(10000, n_steps + 10)
      #  X_train = series[:7000, :n_steps]
    #   #  X_valid = series[7000:9000, :n_steps]
    #   #  X_test = series[9000:, :n_steps]
    #     Y = np.empty((10000, n_steps, 10))
    #     for step_ahead in range(1, 10 + 1):
    #         Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
    #     Y_train = Y[:7000]
    #     Y_valid = Y[7000:9000]
    #     Y_test = Y[9000:]
        
    #     print("X_train, Y_train",X_train.shape, Y_train.shape)
    #     print("n_steps=",n_steps)
        
    #     #np.random.seed(43)
        
    #     #series = generate_time_series(1, 50 + 10)
    #     print("series shape:",series.shape)
    #     X_new, Y_new = series[:, :50, :], series[:, -10:, :]
    #     print("xnew shape",X_new.shape,"Y_new shape",Y_new.shape)
    #     Y_pred = model.predict(X_new)[..., np.newaxis]
    #     print("1Y_pred shape=",Y_pred.shape)
        



    #     print("msx shape=",mat_sales_x.shape)

    #     Y_new=np.empty([mat_sales_x.shape[0],mat_sales_x.shape[1],10])
    #     print("Y new shape",Y_new.shape)
    #     ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
    #  #   print("ys shape",ys.shape)
    
    #   #  pas=predict_ahead_steps   #,dtype=tf.none)
        
    #     for step_ahead in range(1,predict_ahead_steps):
    #           print("\rstep:",step_ahead,"/",predict_ahead_steps,"ys shape",ys.shape,end='\r',flush=True)
    #           y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
    #           ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
    #      #     print("ys",ys,"ys shape",ys.shape,step_ahead)
             
    
    #     print("\rstep:",step_ahead+1,"/",predict_ahead_steps,end='\n\n',flush=True)
    #     pred_product_names=[s + "pred" for s in original_product_names]    #original_product_names
    
    #  #   print("len pred pn=",pred_product_names)
    #  #   print("seresi table=\n",series_table.shape)    
    #  #   print("ys=",ys.shape)
    
    #     series_table,product_name,extended_dates=add_a_new_series(series_table,pred_product_names,ys,start_point,predict_ahead_steps,periods_len)
     
    
    
    # ##########################################################################33
    #     if False:
    #         #print("mat_sales_x shape",mat_sales_x.shape,"n_steps=",n_steps)
              
    #         print("\npredict with MC dropout.  sample_size=",pred_error_sample_size)
             
    #      #   print("Loading mat_sales_x")
    #      #   mat_sales_x=np.load("mat_sales_x.npy")
             
    #        # model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
             
    #                   #   print("mat_sales_x shape",mat_sales_x.shape,"n_steps=",n_steps)
              
             
             
    #         original_steps=mat_sales_x.shape[1]
    #                  # ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
    #                              # print("ys shape",ys.shape)
             
    #         mc_ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
    #                  #  mc_yerr=[0]
    #         mc_yerr=mc_ys/1000   # to get the stddev started with the correct shape
    #                  # print("mc_ys shape",mc_ys.shape)
    #                              # print("mc_yerr=\n",mc_yerr,mc_yerr.shape)
             
             
    #        # pas=predict_ahead_steps   #,dtype=tf.none)
             
    #         for step_ahead in range(1,predict_ahead_steps):
    #            print("\rstep:",step_ahead,"/",predict_ahead_steps,"mc_ys shape",mc_ys.shape,end='\r',flush=True)
    #                    #   y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
              
               
    #            y_probas=np.stack([model(mc_ys[:,:(start_point+step_ahead),:],training=True) for sample in range(pred_error_sample_size)])[:, np.newaxis,:]
    #                        #     print("y_probas shpae=\n",y_probas.shape)
    #            y_mean=y_probas.mean(axis=0)
    #            y_stddev=y_probas.std(axis=0)
             
             
    #                               #     print("ys",ys,"ys shape",ys.shape,step_ahead)
    #                       #     print("y_mean=\n",y_mean,y_mean.shape)
               
               
    #                    #   ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
    #            mc_ys = np.concatenate((mc_ys,y_mean[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
    #            mc_yerr = np.concatenate((mc_yerr,y_stddev[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
             
             
    #         print("\rstep:",step_ahead+1,"/",predict_ahead_steps,end='\n\n',flush=True)
            
            
    #         #########################################33 
               
    #         pred_product_names=[s + "pred_mc" for s in original_product_names]
             
             
    #                    #  series_table=series_table.T
    #         series_table,product_names,extended_dates=add_a_new_series(series_table,pred_product_names,mc_ys,start_point,predict_ahead_steps,periods_len)
            
    #       #  print("\n 2series table=\n",series_table.columns,series_table.shape)
           
    #          #############################################################
             
    #         pred_product_names=[s + "yerr_mc" for s in original_product_names]  
             
             
           
             
    #                   #   series_table=series_table.T    
    #         series_table,product_names,extended_dates=add_a_new_series(series_table,pred_product_names,mc_yerr,start_point,predict_ahead_steps,periods_len)
            
            
    #         print("MC dropout predict finished\n")
            
        
    #  #   print("\n 3series table=\n",series_table.columns,series_table.shape)
    # ##############################################################
    #   #  print("product names=",product_names)
    #  #   series_table=series_table.T
    #     for p in range(0,series_table.shape[0]):
    #     #    plt.figure(figsize=(11,4))
    #         plt.figure(figsize=(14,5))
     
    #         plt.subplot(121)
            
    #         ax = plt.gca()
    #     #    ax.tick_params(axis = 'both', which = 'major',rotation=90, labelsize = 6)
    #         ax.tick_params(axis = 'x', which = 'major',rotation=45, labelsize = 8)
    
    #     #       ax.tick_params(axis = 'both', which = 'minor', rotation=90, labelsize = 6)
    
    #         plt.title("Sales/day:Actual+Pred: "+str(product_names[p]+"_"+str(qnames[model_number])),fontsize=10)
          
    #         plt.ylabel("Units or $ / day",fontsize=9)
    #         plt.xlabel("Date",fontsize=9) 
    #         graph_a_series(series_table,extended_dates,str(product_names[p]))
    #         #,series_table.columns)
            
    #         plt.legend(loc="best",fontsize=8)
    #         if str(product_names[p][-7:])=="yerr_mc":
    #             save_fig("sales_pred_"+str(product_names[p]),c.images_path)
    #         plt.show()
         
            
         
    #    # print("Save graph")
    #    # save_fig("sales_pred")
    
    
    #     for p in range(0,n_query_rows):
    #      #   ax = plt.gca()
    #         plt.plot(range(0,len(extended_dates)), extended_dates,ax=x, markersize=5, label="period")
            
    #         plt.title("Series Pred: Actual vs 95% Prediction: "+str(product_names[p]),fontsize=14)
    #         plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "b-", markersize=5, label="actual")
    #         plt.plot(range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
    #    #     plt.plot(range(start_point,original_steps), mc_ys[0,:(original_steps-start_point),p],"y-", markersize=5, label="mc validation")
    # #        plt.plot(range(mat_sales_x.shape[1]-1,mat_sales_x.shape[1]+predict_ahead_steps-1), mc_ys[0,-(predict_ahead_steps):,p], "m-", markersize=5, label="mc prediction")
    #         plt.errorbar(range(mat_sales_x.shape[1]-1,mat_sales_x.shape[1]+predict_ahead_steps-1), mc_ys[0,-(predict_ahead_steps):,p], yerr=mc_yerr[0,-(predict_ahead_steps):,p]*2,errorevery=20,ecolor='magenta',color='red',linestyle='dotted', label="dropout mean pred 95% conf")
    
    
    #      #   plt.plot(range(mat_sales_x.shape[1]-1,mat_sales_x.shape[1]+predict_ahead_steps-1), ys[0,-(predict_ahead_steps):,p], "r-", markersize=5, label="single prediction")
    #         plt.legend(loc="best")
    #         plt.xlabel("Period")
            
    #         plt.show()
     
    
         
    
    
    
    
           
    #######################################################################################    
        
        print("\nwrite predictions to sales_prediction(.....).CSV file....")
    #    dates.sort()
      
       # print("extended series table=\n",extended_series_table)
        
      #  print("\nseries table shape=",series_table.shape)
    
      #  print("First date=",extended_dates[0])
      #  print("last date=",extended_dates[-1])
     #   with open(c.output_dir+"sales_prediction_"+str(qnames[model_number])+"_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
     #       series_table.T.to_csv(f)  #,line_terminator='rn')
        
    
      
    ###############################################################
    
        print("Saving pickled final table - final_series_table.pkl",series_table.shape)
    
        #    series_table=series_table.T       
        pd.to_pickle(series_table,"final_series_table_"+str(qnames[model_number])+".pkl")
       
      #  with open(c.output_dir+"calc_sales_prediction_"+str(qnames[model_number])+"_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
      #      series_table.to_csv(f)  #,line_terminator='rn')
        
        
      #  print("series_table=\n",series_table)    
      ### Only interested in mc_pred columns
      
     #   print("-10:",series_table.columns.iloc[:,-10:])   #,"mt_pred_mc") 
        st=series_table.filter(like='mt_pred_mc', axis=0).T
        
      #  print("pred_mc only=\n",st)
        
        #st_columns=list(st.columns)
      #  print("sbefore t columns",st_columns)
        #for col in st_columns:
        #   st_columns=st_columns[col.find('@'):]
     
       # print("after st columns",st_columns)
        
       # test3=list(st.columns).split("@") 
      #  print("test3=",st_columns)
    #  print("new st=\n",new_st,new_st.columns)
        
      # series table format is unit sales per day
      #we want to group by and sum by week
        
      # forecast_table = st.resample('W', label='left', loffset=pd.DateOffset(days=1)).sum().div(units_per_ctn).round(1)
        #forecast_table = st.resample('W', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
        forecast_table = st.resample('M', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
       
      #  print("forecast table",forecast_table,forecast_table.shape)
        col_names=list(forecast_table.columns)
      #  print("col names=",col_names)
        new_col_names=[x[:x.find("@")] for x in col_names]
        print("new col names=",new_col_names)
      #  new_col_names_dict=dict(new_col_names)
     #   forecast_table.filter(like=mask_str,axis=0)
    
    #    # cols_dict=dict(cols)  
      #  print("new cols name dict",new_col_names_dict)
    #   #  print("index shape=",np.shape(cols))
    #     flat_column_names = [''.join(col).strip() for col in cols] 
    
    #   #  fcn_dict=dict(flat_column_names)    
    #   #  print("fcn dict",fcn_dict)
    
        rename_dict=dict(zip(col_names, new_col_names))
     #   print("rename dict",rename_dict)
    # #   print("tc=",tc)
    #  #   flat_column_names = [a_tuple[0][level] for a_tuple in np.shape(cols[level])[1] for level in np.shape(cols)[0]]
    #   #  print("fcn=",flat_column_names)
        forecast_table.rename(rename_dict, axis='columns',inplace=True)
        
    
       # forecast_table.columns = pd.MultiIndex.from_product([forecast_table.columns, ['C']])
        #teste=pd.MultiIndex.from_frame(forecast_table.T)  #, names=['state', 'observation'])
     
       # forecast_table.index=forecast_table.index.to_timestamp(freq="D",how="S").dt.strftime("%Y-%m-%d")
        forecast_table.index=forecast_table.index.strftime("%Y-%m-%d")
    
        print("forecast_table=\n",forecast_table)
        s=str(new_col_names[0])
        s = s.replace(',', '_')
        s = s.replace("'", "")
        s = s.replace(" ", "")
    
        # to get over the 256 column limit in excel
        forecast_table.to_csv(c.output_dir+"SCBS_"+str(qnames[model_number])+"_"+s+".csv") 
    
        model_number+=1
        #with pd.ExcelWriter("SCB_"+s+".xls") as writer:  # mode="a" for append
        #    forecast_table.to_excel(writer,sheet_name="Units1")
    #         table2.to_excel(writer,sheet_name="Units2")
    #         table3.to_excel(writer,sheet_name="Units3")
    #         table4.to_excel(writer,sheet_name="Units4")
    #         table5.to_excel(writer,sheet_name="CtnsOfEight5")
    #         table6.to_excel(writer,sheet_name="CtnsOfEight6")
    #         table7.to_excel(writer,sheet_name="CtnsOfEight7")
    #         table8.to_excel(writer,sheet_name="CtnsOfEight8")
    
    #     print("\n\nSales Prediction results from",start_d,"written to spreadsheet",cfg.outxlsfile,"\n\n")
    #     f.write("\n\nSales Prediction results from "+str(start_d)+" written to spreadsheet:"+str(cfg.outxlsfile)+"\n\n")
        
       
    ##############################################################################
    
     
        
    #############################################################################  
        
      
    print("\n\nFinished.")
      
    
    
    return


if __name__ == '__main__':
    main()

        
          
          
          

