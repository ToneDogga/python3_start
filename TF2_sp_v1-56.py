# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:10:04 2020

@author: Anthony Paech 2016
"""


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

print("\n\nSeries prediction tool using a neural network - By Anthony Paech 25/2/20")
print("========================================================================")       

print("Python version:",sys.version)
print("\ntensorflow:",tf.__version__)
print("sklearn:",sklearn.__version__)

# Common imports
import numpy as np
import pandas as pd
import os
import random
import csv
import pickle
from pickle import dump,load
import datetime as dt
from datetime import date

from sklearn.preprocessing import StandardScaler,MinMaxScaler

#import itertools
from natsort import natsorted
import import_constants as ic

print("numpy:",np.__version__)
print("pandas:",pd.__version__)

visible_devices = tf.config.get_visible_devices('GPU') 

print("tf.config.get_visible_devices('GPU'):",visible_devices)
answer=input("Use GPU?")
if answer =="n":

    try: 
      # Disable all GPUS 
      tf.config.set_visible_devices([], 'GPU') 
      visible_devices = tf.config.get_visible_devices() 
      for device in visible_devices: 
        assert device.device_type != 'GPU' 
    except: 
      # Invalid device or cannot modify virtual devices once initialized. 
      pass 
    
    #tf.config.set_visible_devices([], 'GPU') 
    
    print("GPUs disabled")
    
else:
    tf.config.set_visible_devices(visible_devices, 'GPU') 
    print("GPUs enabled")
   
    

if not tf.config.get_visible_devices('GPU'):
#if not tf.test.is_gpu_available():
    print("\nNo GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
  #  if IS_COLAB:
  #      print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
else:
    print("\nSales prediction - GPU detected.")


print("tf.config.get_visible_devices('GPU'):",tf.config.get_visible_devices('GPU'))


# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)



def convert_pivot_table_to_numpy_series(series_table):

#     mat_sales=series_table.iloc[row_no_list,:].to_numpy()
     mat_sales=series_table.to_numpy()

     mat_sales=np.swapaxes(mat_sales,0,1)
     mat_sales=mat_sales[np.newaxis] 
     return mat_sales




def build_mini_batch_input(series,no_of_batches,no_of_steps):
    print("build mini batch 2 series shape",series.shape)
    np.random.seed(42) 
    series_steps_size=series.shape[1]
    
  #  series=np.swapaxes(series,0,2)
          

    random_offsets=np.random.randint(0,series_steps_size-no_of_steps,size=(no_of_batches)).tolist()
  #  print("random_offsets=",random_offsets)   #,random_offset.shape)
 #   single_batch=series[:,:no_of_steps]
  #  new_mini_batch=np.roll(series[:,:no_of_steps+1],random_offsets[0],axis=1).astype(int) 
    new_mini_batch=series[:,random_offsets[0]:random_offsets[0]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)

   # prediction=new_mini_batch[:,-1]
    for i in range(1,no_of_batches):
        temp=new_mini_batch[:,:no_of_steps+1]
        new_mini_batch=series[:,random_offsets[i]:random_offsets[i]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)
    #    prediction=np.concatenate((prediction,new_mini_batch[:,-1]))
        new_mini_batch=np.vstack((temp,new_mini_batch[:,:no_of_steps+1]))
        
        if i%100==0:
            print("\rBatch:",i,"new_mini_batch.shape:",new_mini_batch.shape,flush=True,end="\r")

        
 #   print("new_mini_batch=",new_mini_batch,new_mini_batch.shape)
 #   print("prediction=",prediction)
   # print("\n")        
    return new_mini_batch[:,:no_of_steps,:],new_mini_batch[:,1:no_of_steps+1,:]




def graph_whole_pivot_table(series_table,dates): 
    np.random.seed(43) 
    series_table=series_table.T  
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
    ax = plt.gca()
    cols=list(series_table.columns)
 #   print("1cols=",cols)
    del cols[-1]  # delete reference to period column
  #  print("2cols=",cols)
   
  #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
    #colors = cm.rainbow(np.linspace(0, 2, len(cols)))

    for col in cols:
        color=np.random.rand(len(cols),3)
        series_table.plot(kind='line',x='period',y=col,color=color,ax=ax,fontsize=8)

    plt.legend(loc="best")
    plt.title("All unit sales per day")
    plt.ylabel("Units")
    plt.xlabel("Period") 
    plt.show()
  #  print("graph finished")
    return 
 

def extend_pivot_table(series_table,dates,predict_ahead_steps): 
  #  print("ex dates=\n",dates)
    series_table=series_table.T
    series_table=series_table.reset_index()
    plus_five_years = date.today().year+5
    last_date=dates[-1]
    new_dates1=pd.bdate_range(start=last_date, end='3/24/'+str(plus_five_years))  # usa format
    new_dates2=new_dates1.strftime('%Y-%m-%d').to_list()
  
    extended_series=pd.DataFrame(new_dates2[1:predict_ahead_steps+1],columns=['period'])
    for col in series_table.columns:
        if col=='period':
            pass
        else:
            extended_series[col]=np.nan   # matplot lib wont graph nans
    extended_series['period'] = extended_series['period'].astype('category')
   
    extended_series2=series_table.append(extended_series)   #,ignore_index=True)  #,right_index=True, left_on='period')
    extended_series2.set_index('period', inplace=True)
  
    extended_table3=extended_series2.T
    exdates=extended_series2.index.astype(str).tolist()  #.astype(str)) #strftime("%Y-%m-%d"))
    extended_series=extended_table3.T
    return extended_table3,exdates
 

def find_series_type(series_name):
    return series_name[series_name.find(':')+1:]


    

def graph_a_series(series_table,dates,column_names,series_dict): 

 #   series_dict_elem=series_dict_elem.astype(str)  
    
    series_table=series_table.T  
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
 #   print("\ngraph a series, table.T=\n",series_table,series_table.shape)
    ax = plt.gca()
    cols=list(series_table.columns)
    del cols[-1]  # delete reference to period column
    col_count=0
    for col in cols:
      #  print("find series type",col,"=",find_series_type(col))  
        series_suffix= str(find_series_type(col)) 
        print("series suffix=",series_suffix)
        series_type=str(series_dict[series_suffix])   # name, type of plot, colour
   #     print("series type=\n",series_type,">",series_type)   # name, type of plot, colour
        if (series_suffix=="mt_pred_mc"): # | (series_suffix=="mt_yerr_mc")):
            pred_plot=col
            print("pred_polt=",pred_plot)

        if col in column_names:
#            series_table.plot(kind=series_dict_elem[1],x='period',y=col,color=series_dict_elem[2],ax=ax,fontsize=8)
             #    plt.errorbar('period', series_table[col], yerr=series_table.iloc[col_count+1], data=series_table)
            if series_suffix=="mt_yerr_mc":
                print("\nplotting error bar\n")
                plt.errorbar('period', pred_plot, yerr=col, data=series_table,ecolor=series_type)
 
            else:        
                series_table.plot(kind='line',x='period',y=col,color=series_type,ax=ax,fontsize=8)

        col_count+=1    

    return 
    
  




def plot_learning_curves(title,epochs,loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, np.amax(loss)])
    plt.legend(fontsize=11)
    plt.title(title,fontsize=11)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    return


def plot_log_learning_curves(title,epochs,loss, val_loss):
    ax = plt.gca()
    ax.set_yscale('log')
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, np.amax(loss)])
    plt.legend(fontsize=11)
    plt.title(title,fontsize=11)
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.grid(True)
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



def main():

 
    predict_ahead_steps=ic.predict_ahead_steps   # 120
    epochs_cnn=ic.epochs_cnn   # 1
    epochs_wavenet=ic.epochs_wavenet   # 40
    no_of_batches=ic.no_of_batches  #60000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=ic.batch_length   #10  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
    y_length=ic.y_length    #1
    neurons=ic.neurons     #800
    start_point=ic.start_point   #  20
    pred_error_sample_size=ic.pred_error_sample_size
    
    series_dict=ic.series_dict
    
   # series_dict=dict({"_mt":["_mt","line","blue"],"_pred":["_pred","line","red"],"_mc_pred":["_mc_pred","dotted","green"],"_mc_yerr":["_mc_yerr","dotted","green"]})
    
    

    kernel_size=ic.kernel_size  #4   # for CNN
    strides=ic.strides   #2   # for CNN
    
    
# train validate test split 
    train_percent=ic.train_percent   #0.7
    validate_percent=ic.validate_percent    #0.2
    test_percent=ic.test_percent    #0.1
    
    
    
    

########################################################################    

#    load the sales_trans.xls file
# load the query.xls file
#    this contains all the products, product groups, customers, customer groups, glsets and special price categories you want to     

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors


    answer=input("Rebuild batches?")
    if answer=="y":
 
#    mask="(df['product']=='SJ300')"
        filename=ic.filename    #"NAT-raw310120all.xlsx"
   # #     filename="allsalestrans020218-190320.xlsx"
    
        mats=ic.mats   #[5,65]    # series moving average window periods for each data column to add to series table
      #  col_name_list=["10","14"]
      #  window_size=90
        
     
    #    filename=ic.filename   #"shopsales020218to070320.xlsx"
    
        print("loading data....",filename) 
    #   series2,product_names,periods=load_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
     #   series2,product_names,ddates,table,mat_table,mat_sales,mat_sales_90=load_shop_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
        series_table,dates=ic.load_data(mats,filename,series_dict)    #,col_name_list,window_size)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)



        
        product_names=list(series_table.index) 
        print("\nProduct names, length=",product_names,len(product_names))


   
        
        #  this adds 
        graph_whole_pivot_table(series_table,dates)
        
        extended_series_table,extended_dates=extend_pivot_table(series_table,dates,predict_ahead_steps)
        
      #  graph_whole_pivot_table(extended_series_table,extended_dates)
        
     
        print("Saving pickled table - series_table.pkl")
        pd.to_pickle(series_table,"series_table.pkl")
        pd.to_pickle(extended_series_table,"extended_series_table.pkl")

     
        print("Saving product names")
        np.save("product_names.npy",np.asarray(product_names))
        
        print("Saving dates",len(dates))
        with open('dates.pkl', 'wb') as f:
            pickle.dump(dates,f)   
        with open('extended_dates.pkl', 'wb') as f:
            pickle.dump(extended_dates,f)   
 
        scaled_series=series_table.to_numpy()
   #     scaled_series=series_table.to_numpy()
      
        mat_sales_x=np.swapaxes(scaled_series,0,1)
       # mat_sales_x=np.swapaxes(scaled_series,0,1)
 
        mat_sales_x=mat_sales_x[np.newaxis] 
 
        print("Build batches")
        print("mat sales_x.shape=",mat_sales_x.shape)

       
        X,y=build_mini_batch_input(mat_sales_x,no_of_batches,batch_length)


        print("\n\nSave batches")
        np.save("batch_train_X.npy",X)
        np.save("batch_train_y.npy",y)
 
        print("Saving mat_sales_x")
        np.save("mat_sales_x.npy",mat_sales_x)


    else:
        print("\n\nLoad batches")
        X=np.load("batch_train_X.npy")
        y=np.load("batch_train_y.npy")
       #  series2=np.load("series2.npy")   
        print("loading product_names")
        product_names=list(np.load("product_names.npy"))
       # # dates=list(np.load("periods.npy",allow_pickle=True))
        print("loading dates")
        with open('dates.pkl', 'rb') as f:
             dates = pickle.load(f)   
        with open('extended_dates.pkl', 'rb') as f:
             extended_dates = pickle.load(f)   
           
        print("Loading pivot table")        
        series_table= pd.read_pickle("series_table.pkl")
        extended_series_table= pd.read_pickle("extended_series_table.pkl")

        print("Loading mat_sales_x")
        mat_sales_x=np.load("mat_sales_x.npy")

    
           
    
    print("\n\nseries table loaded shape=",series_table.shape,"\n")  
    print("product names=\n",product_names)
  #  print("series table with date=\n",series_table_with_date)
    print("dates array=\n",len(dates))
    
    n_query_rows=X.shape[2]
    n_steps=X.shape[1]-1
    n_inputs=X.shape[2]
    max_y=np.max(X)
      
    original_product_names=product_names
    
    # for p in range(0,series_table.shape[0]):
    #     plt.figure(figsize=(11,4))
    #     plt.subplot(121)
    #     plt.title("A unit sales series: "+str(original_product_names[p]),fontsize=14)
      
    #     plt.ylabel("Units")
    #     plt.xlabel("Period") 
    #     graph_a_series(series_table,dates,original_product_names[p],series_dict)
        
    #     plt.legend(loc="best")
    #     plt.show()
        

    print("epochs_cnn=",epochs_cnn)
    print("epochs_wavenet=",epochs_wavenet)
   # print("dates=",dates)
    
    print("n_query_rows=",n_query_rows)    
    print("n_steps=",n_steps)
    print("n_inputs=",n_inputs)
    print("predict_ahead_steps=",predict_ahead_steps)
   

    print("max y=",max_y)



 #   print("mini_batches X shape=",X[0],X.shape)  
 #   print("mini_batches y shape=",y[0],y.shape)  
   
    batch_size=X.shape[0]
    print("Batch size=",batch_size)

  #  np.save("batch_train_X.npy",X)
  #  np.save("batch_train_y.npy",y)
   
   
    train_size=int(round(batch_size*train_percent,0))
    validate_size=int(round(batch_size*validate_percent,0))
    test_size=int(round(batch_size*test_percent,0))
  
 #  print("train_size=",train_size)
 #  print("validate_size=",validate_size)
 #  print("test_size=",test_size)
   
    X_train, y_train = X[:train_size, :,:], y[:train_size,-y_length:,:]
    X_valid, y_valid = X[train_size:train_size+validate_size, :,:], y[train_size:train_size+validate_size,-y_length:,:]
    X_test, y_test = X[train_size+validate_size:, :,:], y[train_size+validate_size:,-y_length:,:]
   # X_all, y_all = series2,series2[-1]
       #       #  normalise
#    print("Normalising (L2)...")
#    norm_X_train=tf.keras.utils.normalize(X_train, axis=-1, order=2)
#    norm_y_train=tf.keras.utils.normalize(y_train, axis=-1, order=2)


  # print("\npredict series shape",series.shape)
    print("X_train shape, y_train",X_train.shape, y_train.shape)
    print("X_valid shape, y_valid",X_valid.shape, y_valid.shape)
    print("X_test shape, y_test",X_test.shape, y_test.shape)
   
 #########################################################
    
    answer=input("Retrain model(s)?")
    if answer=="y":
        
        print("\nwavenet")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        layer_count=1    

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[None,n_query_rows]))
        model.add(keras.layers.BatchNormalization())
        for rate in (1,2,4,8) *2:
            
            model.add(keras.layers.Conv1D(filters=neurons, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate))
 
       #     if (layer_count==1) #| (layer_count==3):
           #     model.add(keras.layers.Dropout(rate=0.2))   # calls the MCDropout class defined earlier
        #        model.add(keras.layers.BatchNormalization())
 
            if (layer_count==2): # | (layer_count==3):   
             #    model.add(keras.layers.BatchNormalization())
                 model.add(keras.layers.AlphaDropout(rate=0.2))   # calls the MCDropout class defined earlier            
             #     model.add(keras.layers.MCDropout(rate=0.1))   # calls the MCDropout class defined earlier
        #    model.add(keras.layers.Conv1D(filters=neurons, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate))
 
            layer_count+=1    
        model.add(keras.layers.Conv1D(filters=n_query_rows, kernel_size=1))    
        model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        
    #    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),MyCustomCallback()]
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=380),MyCustomCallback()]
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

        
        history = model.fit(X_train, y_train, epochs=epochs_wavenet, callbacks=callbacks,
                            validation_data=(X_valid, y_valid))
                
            
        print("\nsave model wavenet\n")
        model.save("wavenet_sales_predict_model.h5", include_optimizer=True)
      
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
        plot_log_learning_curves("Log learning curve",epochs_wavenet,history.history["loss"], history.history["val_loss"])
        plt.show()



    else:        
        print("\nload model")
      
    
    print("\nSingle Series Predicting",predict_ahead_steps,"steps ahead.")
 
    print("Loading mat_sales_x")
    mat_sales_x=np.load("mat_sales_x.npy")

    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

    print("mat_sales_x shape",mat_sales_x.shape,"n_steps=",n_steps)
   

    original_steps=mat_sales_x.shape[1]
    ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
    print("ys shape",ys.shape)

    pas=predict_ahead_steps   #,dtype=tf.none)
    
    for step_ahead in range(1,pas):
          print("\rstep:",step_ahead,"/",pas,"ys shape",ys.shape,"n_steps",n_steps,end='\r',flush=True)
          y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
          ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
     #     print("ys",ys,"ys shape",ys.shape,step_ahead)
         

    print("\rstep:",step_ahead+1,"/",pas,end='\n\n',flush=True)
    pred_product_names=[s + "_pred" for s in original_product_names]
    print("pred product names=\n",pred_product_names)
 
    extended_series_table,product_names=ic.add_a_new_series(extended_series_table,pred_product_names,ys)

    

    # for p in range(0,extended_series_table.shape[0]):
    #     plt.figure(figsize=(11,4))
    #     plt.subplot(121)
    #     plt.title("Series Pred: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
      
    #     plt.ylabel("Units")
    #     plt.xlabel("Period") 
    #     graph_a_series(extended_series_table,extended_dates,product_names[p],series_dict)
        
    #     plt.legend(loc="best")
    #     plt.show()
        



    # for p in range(0,n_query_rows):
    #     ax = plt.gca()
    #     plt.plot(range(0,len(extended_dates)), extended_dates,ax=ax, markersize=5, label="period")
 
    #     plt.title("Series Pred: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
    #     plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "b-", markersize=5, label="actual")
    #     plt.plot(range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
    #     plt.plot(range(mat_sales_x.shape[1]-1,mat_sales_x.shape[1]+predict_ahead_steps-1), ys[0,-(predict_ahead_steps):,p], "r-", markersize=5, label="prediction")
    #     plt.legend(loc="best")
    #     plt.xlabel("Period")
        
    #     plt.show()
 

##########################################################################33

    #print("mat_sales_x shape",mat_sales_x.shape,"n_steps=",n_steps)
   
    print("\npredict with MC dropout.  sample_size=",pred_error_sample_size)

    print("Loading mat_sales_x")
    mat_sales_x=np.load("mat_sales_x.npy")

    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

    print("mat_sales_x shape",mat_sales_x.shape,"n_steps=",n_steps)
 


    original_steps=mat_sales_x.shape[1]
   # ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
   # print("ys shape",ys.shape)

    mc_ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
  #  mc_yerr=[0]
    mc_yerr=mc_ys/1000   # to get the stddev started with the correct shape
   # print("mc_ys shape",mc_ys.shape)
   # print("mc_yerr=\n",mc_yerr,mc_yerr.shape)


    pas=predict_ahead_steps   #,dtype=tf.none)
    
    for step_ahead in range(1,pas):
          print("\rstep:",step_ahead,"/",pas,"mc_ys shape",mc_ys.shape,"n_steps",n_steps,end='\r',flush=True)
       #   y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
 
          
          y_probas=np.stack([model(mc_ys[:,:(start_point+step_ahead),:],training=True) for sample in range(pred_error_sample_size)])[:, np.newaxis,:]
     #     print("y_probas shpae=\n",y_probas.shape)
          y_mean=y_probas.mean(axis=0)
          y_stddev=y_probas.std(axis=0)

    
     #     print("ys",ys,"ys shape",ys.shape,step_ahead)
     #     print("y_mean=\n",y_mean,y_mean.shape)
          
          
       #   ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
          mc_ys = np.concatenate((mc_ys,y_mean[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
          mc_yerr = np.concatenate((mc_yerr,y_stddev[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)


    print("\rstep:",step_ahead+1,"/",pas,end='\n\n',flush=True)
   
    pred_product_names=[s + "_pred_mc" for s in original_product_names]
    extended_series_table,product_names=ic.add_a_new_series(extended_series_table,pred_product_names,mc_ys)

    pred_product_names=[s + "_yerr_mc" for s in original_product_names]  
    extended_series_table,product_names=ic.add_a_new_series(extended_series_table,pred_product_names,mc_yerr)


    for p in range(0,extended_series_table.shape[0]):
        plt.figure(figsize=(11,4))
        plt.subplot(121)
        plt.title("Series Pred: Actual vs 95% Prediction: "+str(product_names[p]),fontsize=14)
      
        plt.ylabel("Units")
        plt.xlabel("Period") 
        graph_a_series(extended_series_table,extended_dates,product_names[p],series_dict)
        
        plt.legend(loc="best")
        plt.show()
        





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
 


     
    print("MC dropout predict finished\n")

#############################################################################33

#     print("graphing ys=\n",ys,ys.shape)
#     print("extended series table=\n",extended_series_table,extended_series_table.shape)
#     predict_series=np.swapaxes(ys[0],0,1)
#   #  print("predict series=\n",predict_series,predict_series.shape)
#     extra_series=extended_series_table.iloc[:,:(extended_series_table.shape[1]-predict_series.shape[1])].to_numpy()
#   #  print("extra series=\n",extra_series,extra_series.shape)
#     new_ys=np.hstack((extra_series,predict_series))
#   #  print("new ys series=\n",new_ys,new_ys.shape)
#     tes=extended_series_table.T
#   #  tes.set_index('period',inplace=True)
  
#     pred_product_names=[s + "_pred" for s in product_names]   #pred_product_names=product_names+"_pred"
#   #  print("pred prod names",pred_product_names)
#     tnys=new_ys.T
#     i=0
#     for col in pred_product_names:
#   #      print("col=",col,i)
#         tes[col]=tnys[:,i]
#         i+=1
#     tes=tes.reindex(natsorted(tes.columns),axis=1)   # sort by column name
#     extended_product_names=tes.columns
#     extended_series_table=tes.T
#   #  print("end extended series table=\n",extended_series_table,extended_series_table.shape)
    
# #############################################################################3

#     for p in range(0,extended_series_table.shape[0],2):
#         plt.figure(figsize=(11,4))
#         plt.subplot(121)
#         plt.title("A unit sales prediction: "+str(extended_product_names[p]),fontsize=14)
      
#         plt.ylabel("Units")
#         plt.xlabel("Period") 
#         graph_a_prediction(extended_series_table[p:p+2],extended_dates,p)
#        # graph_a_series(extended_series_table[p+1:p+2],extended_dates,p+1,True)
       
#         plt.legend(loc="best")
#         plt.show()
         
  
    
    
    
    
 ############################################################################3
 #  MCDropout predict
    # create a new testing X batch
    ###################################
    #   measure the models accuracy at different dropout levels
    # print("Test accuracy of predictions")
    # y_values=np.stack([model(X_test,training=True) for sample in range(20)])
    # y_value_mean=y_values.mean(axis=0)
    # y_value_std=y_values.std(axis=0)


       
       
    
#     print("\nMCDropout improved Series Predicting",predict_ahead_steps,"steps ahead.")
    
#  #   print("mat sales x shape=",mat_sales_x.shape)
#     n_steps=mat_sales_x.shape[1]
   
#     model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
#     n_steps=mat_sales_x.shape[1]
 
#  #   print("mat_sales_x shape",mat_sales_x.shape)
#     tnys=tnys[np.newaxis]   
#   #  print("tnys=\n",tnys,tnys.shape)
#   #  print("\n")
#  #   print("mat sales x",mat_sales_x)

#   #  for step_ahead in range(1):
# #          y_probas=np.stack([model(X_test,training=True) for sample in range(3)])
#       #    print("X_test=\n",X_test,X_test.shape) 
#       #    y_est=model(X_test,training=True)
#       #    print("y_est=\n",y_est,y_est.shape)        
 
#     y_probas=np.stack([model(tnys,training=True) for sample in range(pred_error_sample_size)])
#   #  print("y_probas shpae=\n",y_probas.shape)
#     y_mean=y_probas.mean(axis=0)
#     y_stddev=y_probas.std(axis=0)
#   #  print("y_mean=\n",y_mean,y_mean.shape)
#   #  print("y_stddev=\n",y_stddev,y_stddev.shape)

#          # y_est=np.round(model.predict(X_test[:1]),2)
#          # print("y_est=\n",y_est,y_est.shape)
#         #  y_mean=np.round(y_probas[:,:1],2)
#         #  print("y_mean=\n",y_mean,y_mean.shape)
#     #      print("sahead loop mat sales 90[0,:]=",mat_sales_90[0,:],mat_sales_90.shape)
#        #   print("sahead loop y mean[:,-1]=",y_mean[:,-1],y_mean.shape)
#        #   mat_sales_x=np.append(mat_sales_x[0,:],y_mean[:,-1]).reshape(1,-1,n_query_rows)
#        #   print("saehad loop mat sales 90=",mat_sales_x.shape,"n steps=",n_steps)
#        #   X_test,y=build_mini_batch_input(mat_sales_x[-n_steps:],10000,batch_length)
#           #X_test=np.stack(X_test,new_X_test)
#       #    print("step ahead=",step_ahead,"y_mean=\n",y_mean[:,-1,:],"y_mean shape=",y_mean.shape)  #[:, np.newaxis,:]
#        #   print("sahead",step_ahead,"sahead loop appended X test shape",X_test.shape)
 
#   #  yerr = np.linspace(0, 0, predict_ahead_steps)
#   #  print("yerr shape=",yerr.shape)
    
        
#     # for p in range(0,tnys.shape[2]):
#     #     plt.figure(figsize=(11,4))
#     #     plt.subplot(121)
#     #     plt.title("Single prediction vs mean of dropout pred: "+str(extended_product_names[p]),fontsize=14)
      
#     #     plt.ylabel("Units")
#     #     plt.xlabel("Period") 
#     #  #   graph_a_prediction(extended_series_table[p:p+2],extended_dates,p)
#     #    # graph_a_series(extended_series_table[p+1:p+2],extended_dates,p+1,True)
#     #     plt.plot(range(0,tnys.shape[1]), tnys[0,:,p], "r-", markersize=5, label="single pred")
#     #     #plt.plot(range(0,tnys.shape[1]), y_mean[0,:,p], "b.", markersize=5, label="dropout mean pred")
#     #     plt.errorbar(range(0,tnys.shape[1]), y_mean[0,:,p], yerr=y_stddev[0,:,p]*2,linestyle='dotted', label="dropout mean pred 95% conf")
 
#     #    # plt.errorbar(range(mat_sales_x.shape[1],mat, ys[0,-(predict_ahead_steps):,p], yerr=yerr, linestyle="-",label='Prediction + error')      

#     #     plt.legend(loc="best")
#     #     plt.show()
         
    
# #################################################################
# # add error bars to extended series table
#     #tes=extended_series_table.T
        







# ##########################################################
#     # i=0
#     # for p in range(0,extended_series_table.shape[0],2):
#     #     plt.figure(figsize=(11,4))
#     #     plt.subplot(121)
#     #     plt.title("A unit sales prediction: "+str(extended_product_names[p]),fontsize=14)
      
#     #     plt.ylabel("Units")
#     #     plt.xlabel("Period") 
#     #     graph_a_prediction(extended_series_table[p:p+1],extended_dates,p)
#     #    # graph_a_series(extended_series_table[p+1:p+2],extended_dates,p+1,True)
#     #     plt.errorbar(range(0,tnys.shape[1]), y_mean[0,:,i], yerr=y_stddev[0,:,i]*2,linestyle='dotted', label="dropout mean pred 95% conf")
#     #     i+=1
#     #     plt.legend(loc="best")
#     #     plt.show()
        
#  ##############################################################


#     for p in range(0,n_query_rows):
#         plt.title("Series Pred: Actual vs 95% conf Prediction: "+str(product_names[p]),fontsize=14)
#         plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "b-", markersize=5, label="actual")
#         plt.plot(range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
#         plt.plot(range(mat_sales_x.shape[1]-1,mat_sales_x.shape[1]+predict_ahead_steps-1), ys[0,-(predict_ahead_steps):,p], "r-", markersize=5, label="prediction")
#         plt.errorbar(range(0,tnys.shape[1]), y_mean[0,:,p], yerr=y_stddev[0,:,p]*2,linestyle='dotted', color='yellow',label="dropout mean pred 95% conf")

#         plt.legend(loc="best")
#         plt.xlabel("Period")
        
#         plt.show()
    
    

       
#######################################################################################    
    
    print("\nwrite predictions to sales_prediction.CSV file....")
#    dates.sort()
    extended_series_table=extended_series_table.T
    print("extended series table=\n",extended_series_table)
    
    print("\nextended series table shape=",extended_series_table.shape)

    print("First date=",extended_dates[0])
    print("last date=",extended_dates[-1])
    with open("sales_prediction_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
        extended_series_table.to_csv(f)  #,line_terminator='rn')
    
    print("\n\nFinished.")
     
    return


if __name__ == '__main__':
    main()


    
          
          
          
