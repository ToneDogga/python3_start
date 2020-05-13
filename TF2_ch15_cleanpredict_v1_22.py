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
    
 
  
    
def load_series(start_point,end_point):    
    with open("batch_dict.pkl", "rb") as f:
         seriesbatches = pickle.load(f)
    #mat_sales_x =seriesbatches[0][7]
    series=seriesbatches[0][9]
    print("full series shape=",series.shape)
    mat_sales_x=series.to_numpy().astype(np.int32)
   # print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))
  #  mat_sales_x=mat_sales_x.astype(np.int32)

    #Wmat_sales_x=mat_sales_x[...,np.newaxis].astype(np.int32)
    print("mat sales x.shape",mat_sales_x.shape)
    print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))

    print("loaded mat_sales x shape",mat_sales_x.shape)
    print("start point=",start_point)
    print("end_point=",end_point)
    shortened_series=series.iloc[:,start_point:-1]
    mat_sales_x=mat_sales_x[:,start_point:end_point+1][...,np.newaxis]
    print("trimmed mat_sales x shape",mat_sales_x.shape)
    print("batch len=",batch_length)
    print("shortened_series=",shortened_series.shape)
   # series=series[:,start_point:end_point]
    #print("series trimmed=",series,series.shape)
    dates=shortened_series.T.index.astype(str).tolist()  #.astype(str)) #strftime("%Y-%m-%d"))
    shortened_series=shortened_series.to_numpy()
    shortened_series=shortened_series[...,np.newaxis].astype(np.int32)
    return shortened_series,mat_sales_x,dates   #[..., np.newaxis].astype(np.float32)    
  

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
   
    return X   #,new_batches[:,1:batch_length+1,:]




# def create_Y(shortened_series,X,batch_length,pred_length):   #,start_point,end_point):
#    # print("Y total_existing_steps",total_existing_steps)
 
# #   batch_length=X.shape[1]
# #    n_inputs=X.shape[2]
# #    print("X batch length=",batch_length)
#   #  print("start point=",start_point)
#   #  print("end point",end_point)
#   #  pred_length=end_point-start_point
#     print("pred length",pred_length)
#  #   Y_window_length=(end_point-start_point)-pred_length
#  #   print("Y window length=",Y_window_length)
#     Y = np.empty((X.shape[0], sample_length, _length),dtype=np.int32)
#   #  Y = np.empty((X.shape[0], batch_length, pred_length))
 
#     print("new Y shape",Y.shape)
#     for step_ahead in range(1, sample_length + 1):
#         Y[:,:,step_ahead - 1] = shortened_series[:, step_ahead:step_ahead+sample_length,0]  #,n_inputs-1]  #+1
#   #      print("step a=",step_ahead,"X=",X[..., step_ahead:step_ahead+batch_length,0],"ss=",shortened_series[..., step_ahead:step_ahead+batch_length+1, 0])  #+1

#     print("final create Y.shape=",Y.shape)

#     return Y




# def plot_series(series, y=None, y_pred=None, x_label="$date$", y_label="$units/day$"):
#     plt.plot(series, "-")
#     if y is not None:
#         plt.plot(n_steps, y, "b-")   #, markersize=10)
#     if y_pred is not None:
#         plt.plot(n_steps, y_pred, "r-")
#     plt.grid(True)
#     if x_label:
#         plt.xlabel(x_label, fontsize=16)
#     if y_label:
#         plt.ylabel(y_label, fontsize=16, rotation=90)
#     plt.hlines(0, 0,  series.shape[0], linewidth=1)
#   #  plt.axis([0, n_steps + 1, -1, 1])
#     plt.axis([0, series.shape[0], 0 , np.max(series)])





def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
 
    
 
def plot_learning_curves(loss, val_loss,title):
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


def create_plot_df(plot_dict):
# pass the plot dictionary to a pandas dataframe and return
 #   print("plot_dict=\n",plot_dict)
    
 #   date_len=len(plot_dict[(0,0,"dates")])
    date_len=1300
 #   print("date len",date_len)
  #  first_date="02/02/18".strftime('%Y-%m-%d')
 #   last_date=series_table.index[-1].strftime('%Y-%m-%d')
  #  series_table=series_table.T
   # if series_table.index.nlevels>=1:
   #     series_table.columns=series_table.columns.astype(str)
 #   product_names=series_table.columns.astype(str)
 #   product_names=series_table.columns   #[1:]   #.astype(str)

 #   print("xtend series table",product_names)
 #   len_product_names=len(product_names)
 #   print("len pn=",len_product_names)
    dates = pd.period_range("02/02/18", periods=date_len)   # 2000 days
 #   print("series_table=\n",series_table,series_table.index)
 #   new_table = pd.DataFrame(np.nan, index=product_names,columns=pidx)   #,dtype='category')  #series_table.columns)
    date_len=len(dates)
    print("date len",date_len)
   # series_names=["dates"]
    start_points=[]
   # data_lengths=[date_len]
   # new_data_lengths=[date_len]
  #  data=[plot_dict[(0,0,"dates")]]
    #data=[plot_dict[(0,0,"dates")]]
   
    for series_number in range(1,9):        
        subdict = {k: v for k, v in plot_dict.items() if str(k).startswith("("+str(series_number))}
        if subdict: 
            for series_type in range(1,9):
                   subdict2 = {k: v for k, v in subdict.items() if str(k).startswith("("+str(series_number)+", "+str(series_type))}
                   if subdict2:
                       key_value=list(subdict2.keys())[0]
                       dict_value=list(subdict2.values())[0]
                       dict_name=key_value[2]
                       print("dict name=",dict_name)
                       print("dict value=",dict_value)
                       print("start points[]",start_points)
                       if series_number==1:   # start_point
                          # sp=dict_value
                           start_points.append(dict_value)   
                        #   series_names.append(dict_name)
                       elif series_number==2:   # data length
                      #     data_lengths.append(np.max(dict_value.shape))
                        #   data_lengths.append(np.max(dict_value.shape))
                        #   print("data len",data_lengths)

                           sp=start_points.pop(0)
                           print("sp",sp)
                           #data.append(dict_value)
                           filler_array=np.zeros(sp)
                           filler_array[:] = np.nan
                           back_filler=date_len-(sp+dict_value.shape[0])
                           
                        #   print("data=",subdict2.values())
                        #   print("key value=",key_value[-1])
                        #   kv=key_value[-1]

                           if back_filler<=0:
                               back_filler_array=np.zeros(0)
                           else:    
                               back_filler_array=np.zeros(back_filler)
                           back_filler_array[:] = np.nan
    
                               
                               
                           subdict3 = {k: v for k, v in plot_dict.items() if str(k).startswith(("("+str(series_number)+", "+str(series_type)))}
    
                           kv=list(subdict3.keys())[0] 
                           print("key v=",kv)
                           uk={kv:np.concatenate((filler_array,plot_dict[kv],back_filler_array),axis=0)[:date_len]}
      
                           plot_dict.update(uk)
                           try:
                               print(kv,"pds=",len(plot_dict[kv]))
                           except:
                               pass
                         #  print("ploy dict[kv] after concat shape=",plot_dict[kv].shape)   
                        #   new_data_lengths.append(plot_dict[kv].shape[0])
    
 
# plot only array fdata
    series_number=2        
    subdict = {k[2]: v for k, v in plot_dict.items() if str(k).startswith("("+str(series_number))}
 
    plot_df=pd.DataFrame.from_dict(subdict,orient='columns',dtype=np.int32)
    plot_df.index=dates

    print("plot_df=\n",plot_df)
    

    return plot_df



  
 
    # plot_dict=dict({(1,1,"Actual_start") : start_point,
    #                 (2,1,"Actual_data") : predict_values[0,:,0],
    #                 (1,2,"MC_predict_mean_start") : end_point,
    #                 (2,2,"MC_predict_mean_data") : Y_mean,
    #                 (1,3,"MC_predict_stddev_start") : end_point,
    #                 (2,3,"MC_predict_stddev_data") : Y_stddev,
    #                 (0,0,"dates") : dates
    #                 })
 
  
 


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





###########################################3
np.random.seed(42)

patience=10
epochs=12
neurons=1000   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
start_point=32
end_point=800
# predict aherad length is inside batch_length
sample_length=end_point-start_point
predict_ahead_steps=365   #int(round(sample_length/2,0))
batch_ahead_steps=3

pred_error_sample_size=50

#batch_length=20+predict_ahead_length
#batch_length=(end_point-start_point)+predict_ahead_length
#X_window_length=batch_length-predict_ahead_length
#future_steps=400
#blank_future_days=365
# batch_total=100000

no_of_batches=10000
batch_length=predict_ahead_steps

train_percent=0.7
validate_percent=0.2
test_percent=0.1

n_train=int(round((no_of_batches*train_percent),0))
n_validate=int(round((no_of_batches*validate_percent),0))
n_test=int(round((no_of_batches*test_percent),0))



#  Now let's create an RNN that predicts the next 10 steps at each time step. 
# That is, instead of just forecasting time steps 50 to 59 based on time steps 0 to 49,
#  it will forecast time steps 1 to 10 at time step 0, then time steps 2 to 11 at time 
# step 1, and so on, and finally it will forecast time steps 50 to 59 at the last time step.
#  Notice that the model is causal: when it makes predictions at any time step, 
# it can only see past time steps.

#print("cretae an RNN that predicts the next 10 steps at each time step")



#n_steps = 50
shortened_series,mat_sales_x,dates = load_series(start_point,end_point)
print("shoerened series.shape=",shortened_series.shape)
print("len dates=",len(dates))
#print("mat_sales_x [:,2:]=",mat_sales_x[:,1:].shape)
#print("mat_sales_x[:,:-1]=",mat_sales_x[:,:-1].shape)


X=create_batches(no_of_batches,batch_length,mat_sales_x[:,:-1],start_point,end_point)
#print("X.shape",X[0],X.shape)
print("X size=",X.nbytes,"bytes")


X_train = X[:n_train]
X_valid = X[n_train:n_train+n_validate]
X_test = X[n_train+n_validate:]



#Y=create_batches(no_of_batches,batch_length,mat_sales_x[:,1:],start_point,end_point)
#print("start_Y.shape",start_Y[0],start_Y.shape)#

Y = np.empty((no_of_batches, batch_length,predict_ahead_steps))
 
print("new Y shape",Y.shape)
for step_ahead in range(1, predict_ahead_steps + 1):
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
print("sample length",sample_length)

print("X_train",X_train.shape)
print("Y_train",Y_train.shape)
print("X_valid",X_valid.shape)
print("Y_valid",Y_valid.shape)
print("X_test",X_test.shape)
print("Y_test",Y_test.shape)

#print("X_train[0]=\n",X_train[0])
#print("\nY_train[0]=\n",Y_train[0])
##############################################################
answer=input("load saved model?")
if answer=="y":
    print("\n\nloading model...")  
    model=keras.models.load_model("GRU_Dropout_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
else:
    
    
    print("GRU with dropout")
    
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model = keras.models.Sequential([
        keras.layers.GRU(neurons, return_sequences=True, input_shape=[None, n_inputs]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.BatchNormalization(),
        keras.layers.GRU(neurons, return_sequences=True),
        keras.layers.Dropout(rate=0.2),
        keras.layers.BatchNormalization(),
        keras.layers.TimeDistributed(keras.layers.Dense(predict_ahead_steps))
    ])
    
    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),MyCustomCallback()]
    
    history = model.fit(X_train, Y_train, epochs=epochs,
                        validation_data=(X_valid, Y_valid))
    
    
    print("\nsave model\n")
    model.save("GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
       
    model.summary()

    plot_learning_curves(history.history["loss"], history.history["val_loss"],"GRU and dropout")
    plt.show()


###################################3

    
# mc_ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
#           #  mc_yerr=[0]
# mc_yerr=mc_ys/1000   # to get the stddev started with the correct shape
#           # print("mc_ys shape",mc_ys.shape)
#                       # print("mc_yerr=\n",mc_yerr,mc_yerr.shape)
  
 



print("\nPredicting....")
#predict = np.empty((X.shape[0], batch_length, pred_length),dtype=np.int32)
predict_values=mat_sales_x.astype(np.float32)   #[0,:,0]     #]

Y_probas = np.empty((1,batch_length))  #predict_ahead_steps))
#Y = np.empty((no_of_batches, batch_length,predict_ahead_steps))
    
#mc_ys= np.empty((1))  #predict_values[0,-1:,0]    #model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
          #  mc_yerr=[0]
#mc_yerr=mc_ys/1000   # to get the stddev started with the correct shape
          # print("mc_ys shape",mc_ys.shape)
                      # print("mc_yerr=\n",mc_yerr,mc_yerr.shape)
  
#yprint("mc_ys shape",mc_ys.shape)

 

for batch_ahead in range(0,1): #predict_ahead_steps*batch_ahead_steps,predict_ahead_steps):
    
    #    print("step ahead=",step_ahead)
  #  print("2predict_values.shape=",predict_values.shape)
  #  print("batch ahead=",batch_ahead)
    Y_pred=model.predict(predict_values[:,predict_ahead_steps:])    #[:,batch_ahead:])
    
  #  print("predicted  Y_pred[0]=",Y_pred.shape)
    
    Y_diag=Y_pred[0,:,-1][np.newaxis,...]
    Y_diag=Y_diag[...,np.newaxis]
    
    plt.plot(np.arange(start_point,start_point+predict_values.shape[1]), predict_values[0, :, 0], "b-", label="Actual",markersize=4)
 #   plt.plot(np.arange(end_point+batch_ahead,end_point+batch_ahead+Y_diag.shape[1]), Y_diag[0, :, 0], "r-", label="Predict",markersize=4)
    
######################################################    
    
    
  #  y_probas=np.stack([model(predict_values[:,predict_ahead_steps:],training=True) 
    for sample in range(pred_error_sample_size):                  
#         y_probas=np.stack(model(predict_values[:,predict_ahead_steps:],training=True))[:, np.newaxis,:]           
        Y_probs=model(predict_values[:,predict_ahead_steps:],training=True)[np.newaxis,...]          
     #   print(sample,"Y_probs[0,:,-1].shape",Y_probs[0,:,-1].shape)
        # Y_diag=Y_probas[0,:,-1][np.newaxis,...]
        new_probs=Y_probs[0,:,-1]
      #  print("new_probs shpae=\n",new_probs.shape)
        Y_probas=np.concatenate((Y_probas,new_probs),axis=0)  #[np.newaxis,...]
    

   # print("Y_probas shpae=\n",Y_probas,Y_probas.shape)
    Y_probas=np.nan_to_num(Y_probas,nan=0, posinf=np.nan, neginf=np.nan)
    Y_mean=Y_probas.mean(axis=0)##[np.newaxis]
    Y_stddev=Y_probas.std(axis=0)#[np.newaxis]
  
    #print("ys",ys,"ys shape",ys.shape,step_ahead)
   # print("Y_mean=\n",Y_mean,Y_mean.shape)
   # print("Y_stddev=\n",Y_stddev,Y_stddev.shape)
   
   # print("1mc_ys=",mc_ys,mc_ys.shape)
    
            #   ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
   # mc_ys = np.concatenate((mc_ys,Y_mean),axis=0)    #[:, np.newaxis,:]), axis=1)
   # mc_yerr = np.concatenate((mc_yerr,Y_stddev),axis=0)    #[:, np.newaxis,:]), axis=1)

 #   plt.plot(np.arange(end_point+batch_ahead,end_point+batch_ahead+Y_diag.shape[1]), Y_diag[0, :, 0], "r-", label="Errorbar",markersize=4)
#    plt.errorbar(np.arange(end_point+batch_ahead,end_point+batch_ahead+Y_diag.shape[1]), mc_ys, yerr=mc_yerr, fmt="r.",ms=3,data=series_table,ecolor="magenta",errorevery=1)
   # print("2mc_ys=",mc_ys,mc_ys.shape)

    predict_values=np.concatenate([predict_values,Y_diag],axis=1)

   # plt.plot(np.arange(start_point,start_point+predict_values.shape[1]), predict_values[0, :, 0], "b-", label="Actual",markersize=4)
   #  plt.plot(np.arange(end_point+batch_ahead,end_point+batch_ahead+Y_diag.shape[1]), Y_diag[0, :, 0], "r-", label="Predict",markersize=4)
 
 
    # dictionary mat type code :   aggsum field, name, color
       # mat_type_dict=dict({"u":["qty","units","b-"]
                      #  "d":["salesval","dollars","r-"],
                      #  "m":["margin","margin","m."]
    #                   })
         
 
    
 
    plot_dict=dict({(1,1,"Actual_start") : start_point,
                    (2,1,"Actual_data") : predict_values[0,:,0],
                    (1,2,"MC_predict_mean_start") : end_point,
                    (2,2,"MC_predict_mean_data") : Y_mean,
              #      (1,3,"MC_predict_stddev_start") : end_point,
              #      (2,3,"MC_predict_stddev_data") : Y_stddev,
                    (0,0,"dates") : dates
                    })
 
                   
    #print("plot_dict=",plot_dict) 
    plot_df=create_plot_df(plot_dict)

    plot_df.plot()
   # plt.plot(np.arange(end_point+batch_ahead,end_point+batch_ahead+Y_mean.shape[0]), Y_mean, "g-", label="MC Predict",markersize=4)
    
    


  #  plt.errorbar(np.arange(end_point+batch_ahead,end_point+batch_ahead+Y_diag.shape[1]), mc_ys, yerr=mc_yerr, fmt="r.",label="Errorbar",ms=3,ecolor="magenta",errorevery=1)



 #   plt.plot(np.arange(start_point,start_point+mat_sales_x.shape[1]), mat_sales_x[0, :, 0], "b-", label="Actual",markersize=4)
     
 #   plt.plot(np.arange(start_point,start_point+shortened_series.shape[1]), shortened_series[0, :, 0], "b-", label="Actual",markersize=4)
   # plt.plot(np.arange(end_point,end_point+Y_diag.shape[1]), Y_diag[0, :, 0], "r-", label="Predict",markersize=4)
   # plt.plot(np.arange(start_point,start_point+predict_values.shape[1]), predict_values[0, :, 0], "r-", label="Actual+Predict2",markersize=4)
    
    #plt.plot(np.arange(start_point,start_point+predict_values.shape[1]), predict_values[0, :, 0], "b-", label="Actual",markersize=1)
    plt.legend(fontsize=14)
    plt.title("Unit sales prediction")
    plt.xlabel("Days")
    plt.ylabel("units")
    plt.grid(True)
    
    
    plt.show()     
    
 
# pas=predict_ahead_steps   #,dtype=tf.none)
  
# for step_ahead in range(1,predict_ahead_steps):
#     print("\rstep:",step_ahead,"/",predict_ahead_steps,"mc_ys shape",mc_ys.shape,end='\r',flush=True)
#             #   y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
   
    
#     y_probas=np.stack([model(mc_ys[:,:(start_point+step_ahead),:],training=True) for sample in range(pred_error_sample_size)])[:, np.newaxis,:]
#                 #     print("y_probas shpae=\n",y_probas.shape)
#     y_mean=y_probas.mean(axis=0)
#     y_stddev=y_probas.std(axis=0)
  
  
#                        #     print("ys",ys,"ys shape",ys.shape,step_ahead)
#                #     print("y_mean=\n",y_mean,y_mean.shape)
    
    
#             #   ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
#     mc_ys = np.concatenate((mc_ys,y_mean[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
#     mc_yerr = np.concatenate((mc_yerr,y_stddev[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
  
  
# print("\rstep:",step_ahead+1,"/",predict_ahead_steps,end='\n\n',flush=True)
 




#predict_ahead("GRU with dropout predictions",model,mat_sales_orig,mat_sales_pred,mat_sales_x,X,batch_length,X_window_length,predict_ahead_length,start_point,end_point)

