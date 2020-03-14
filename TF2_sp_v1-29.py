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

# try:
#     # %tensorflow_version only exists in Colab.
#     %tensorflow_version 2.x
#     IS_COLAB = True
# except Exception:
#     IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

print("\n\nSales prediction tool using a neural network - By Anthony Paech 25/2/20")
print("=======================================================================")       

print("Python version:",sys.version)
print("\ntensorflow:",tf.__version__)
print("sklearn:",sklearn.__version__)

# Common imports
import numpy as np
import pandas as pd
import os
import random
import csv

print("numpy:",np.__version__)
print("pandas:",pd.__version__)
# print("tf.config.list_physical_devices('GPU')",tf.config.list_physical_devices('GPU'))

# if not tf.config.list_physical_devices('GPU'):
#     print("\nNo GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
#   #  if IS_COLAB:
#   #      print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
# else:
#     print("\nSales prediction - GPU detected.")

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # disables the GPU
# Set CPU as available physical device


#my_devices = tf.config.list_physical_devices(device_type='CPU')
#tf.config.set_visible_devices(devices= my_devices, device_type='CPU')

#print("tf.config.list_physical_devices('CPU'):",tf.config.list_physical_devices('CPU'))
#print("tf.config.list_physical_devices('GPU'):",tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU') 
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

print("tf.config.get_visible_devices('GPU'):",tf.config.get_visible_devices('GPU'))
print("GPUs disabled")

if not tf.config.get_visible_devices('GPU'):
#if not tf.test.is_gpu_available():
    print("\nNo GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
  #  if IS_COLAB:
  #      print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
else:
    print("\nSales prediction - GPU detected.")




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

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

#n_steps = 50

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
    



   
def load_data(filename):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows


    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

  #  df["period"]=df.date.dt.to_period('W')
  #  df["period"]=df.date.dt.to_period('B')    # business days  'D' is every day
    df["period"]=df.date.dt.to_period('D')    # business days  'D' is every day

   
 #   mask = mask.replace('"','').strip()    
 #   print("mask=",mask)
    
 #   mask=((df['code']=='FLPAS') & (df['product']=='SJ300'))

  #  mask=(df['product']=='SJ300')
  #  mask=(df['code']=='FLPAS')
 #   mask=((df['code']=='FLPAS') & (df['product']=="SJ300") & (df['glset']=="NAT"))
 #   mask=((df['productgroup']==10))
  
    mask=((df['code']=='FLPAS') & ((df['product']=="CAR280") | (df['product']=="SJ300")))
  #  mask=((df['code']=='FLPAS') & (df['productgroup']==10))  # & ((df['product']=='SJ300') | (df['product']=='AJ300')))
 #   mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='AJ300') | (df['product']=='TS300')))

   # print("mask=",str(mask))
    print("pivot table being created.")
  #  table = pd.pivot_table(df[mask], values='qty', index=['code','productgroup','product'],columns=['week'], aggfunc=np.sum, margins=False,observed=False, fill_value=0)   #observed=True
 #   table = pd.pivot_table(df[mask], values='qty', index=['code','product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
    table = pd.pivot_table(df[mask], values=['qty'], index=['code','product'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
 
    print("\ntable=\n",table)   #.head(5))#  #   f.write("\n\n"+table.to_string())
    print("table created.")
    
    product_names=list(table.index)   #"t"  #list(table[table["product"]])
    #print("product names=",product_names)
#    table.drop(columns=['glset'],axis=0)
    sales=table.to_numpy()
 #   sales=sales[...,:n_steps]
    print("sales=\n",sales,sales.shape)        
    return sales[..., np.newaxis].astype(np.float32),product_names




def load_shop_data(filename):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows


    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

  #  df["period"]=df.date.dt.to_period('W')
  #  df["period"]=df.date.dt.to_period('B')    # business days  'D' is every day
    df["period"]=df.date.dt.to_period('W')    # business days  'D' is every day

   
 #   mask = mask.replace('"','').strip()    
 #   print("mask=",mask)
    
 #   mask=((df['code']=='FLPAS') & (df['product']=='SJ300'))

  #  mask=(df['product']=='SJ300')
  #  mask=(df['code']=='FLPAS')
 #   mask=((df['code']=='FLPAS') & (df['product']=="SJ300") & (df['glset']=="NAT"))
 #   mask=((df['productgroup']==10))
  
  #  mask=((df['code']=='FLPAS') & ((df['product']=="CAR280") | (df['product']=="SJ300")))
  #  mask=((df['code']=='FLPAS') & (df['productgroup']==10))  # & ((df['product']=='SJ300') | (df['product']=='AJ300')))
 #   mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='AJ300') | (df['product']=='TS300')))

   # print("mask=",str(mask))
    print("pivot table being created.")
  #  table = pd.pivot_table(df[mask], values='qty', index=['code','productgroup','product'],columns=['week'], aggfunc=np.sum, margins=False,observed=False, fill_value=0)   #observed=True
 #   table = pd.pivot_table(df[mask], values='qty', index=['code','product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
#    table = pd.pivot_table(df[mask], values=['qty'], index=['code','product'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
    table = pd.pivot_table(df, values=['salesval'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False, fill_value=0)   #observed=True
 
    print("\ntable=\n",table)   #.head(5))#  #   f.write("\n\n"+table.to_string())
    print("table created.")
    
    product_names=[""]  #list(table.index)   #"t"  #list(table[table["product"]])
    #print("product names=",product_names)
#    table.drop(columns=['glset'],axis=0)
    sales=table.to_numpy()
 #   sales=sales[...,:n_steps]
    print("sales=\n",sales,sales.shape)        
    return sales[..., np.newaxis].astype(np.float32),product_names





def build_mini_batch_input(series,no_of_batches,no_of_steps):
    print("build mini batch series shape",series.shape)
    np.random.seed(42) 
    series_steps_size=series.shape[1]
  #  print("series steps size=",series_steps_size)

#  input series array is structured as a 2d array [[step values]]  shape eg (1,82)
# the numnber of mini batches should be the same length as the input layer number of neurons
# the output array is structured as a 3D array [no of mini-batches,steps,step values]
# each mini batch is a training series of say 20 steps long, with a random starting point somewhere on the batch size.
# if the starting point cannot go too close to the end of the series as it would
# overflow over the end.
# also how does it predict the first 20 steps?
    
    
    # series shape is (2,91,1) say
    # that is n_query_rows=2
    # n_steps=91
    # n_inputs=1
    
    
    # I need to change to   series shape (1,91,2)
    # n_query_rows=1
    # n_steps=91
    # n_inputs =2 
 #   print("series shape before change:",series[0,:10],series.shape)
    
    series=np.swapaxes(series,0,2)
          
  #  print("series shape after change:",series[0,:10],series.shape)

    random_offsets=np.random.randint(0,series_steps_size-no_of_steps,size=(no_of_batches)).tolist()
  #  print("random_offsets=",random_offsets)   #,random_offset.shape)
 #   single_batch=series[:,:no_of_steps]
  #  new_mini_batch=np.roll(series[:,:no_of_steps+1],random_offsets[0],axis=1).astype(int) 
    new_mini_batch=series[:,random_offsets[0]:random_offsets[0]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)

    prediction=new_mini_batch[:,-1]
    for i in range(1,no_of_batches):
        temp=new_mini_batch[:,:no_of_steps]
        new_mini_batch=series[:,random_offsets[i]:random_offsets[i]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)
        prediction=np.concatenate((prediction,new_mini_batch[:,-1]))
        new_mini_batch=np.vstack((temp,new_mini_batch[:,:no_of_steps]))
        
        if i%100==0:
            print("\rBatch:",i,"new_mini_batch.shape:",new_mini_batch.shape,flush=True,end="\r")

        
 #   print("new_mini_batch=",new_mini_batch,new_mini_batch.shape)
 #   print("prediction=",prediction)
    print("\n")        
    return new_mini_batch[:,:no_of_steps],prediction 





def build_mini_batch_input2(series,no_of_batches,no_of_steps):
    print("build mini batch 2 series shape",series.shape)
    np.random.seed(42) 
    series_steps_size=series.shape[1]
    
    series=np.swapaxes(series,0,2)
          

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




def plot_learning_curves(title,epochs,loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, np.amax(loss)/3])
    plt.legend(fontsize=11)
    plt.title(title,fontsize=11)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    return


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


#  How the batching works
#  The input array is three dimentional
# batch_size, n_steps, n_inputs
#
# batch size is the number of batches of each instance of the series
# in other words, it is the number of unique training series (1)
#
# n_steps is the number of time steps in each batch (104)
#
# n_inputs is the actual size of the data at each time step  (1)   

#n_steps = 50



def main():
#    print("\n\nSales prediction tool using a neural network - By Anthony Paech 25/2/20")
#    print("=======================================================================")       
 
 #   mask=((df["code"]=="FLPAS") & (df["product"]=="SJ300"))

   # n_steps = 100
    predict_ahead_steps=80 #4
    epochs=100
    no_of_batches=50000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=100
    
    #104 #364 #30   #20
 #   no_of_batch_copies=4  # duplicate the batches
 #   max_batch_size=1
    
   # train validate test split
    train_percent=0.7
    validate_percent=0.2
    test_percent=0.1

########################################################################    

#    load the sales_trans.xls file
# load the query.xls file
#    this contains all the products, product groups, customers, customer groups, glsets and special price categories you want to     






     
    answer=input("Rebuild batches?")
    if answer=="y":
 

#    mask="(df['product']=='SJ300')"
   #     filename="NAT-raw310120all.xlsx"
       # filename="cashsales020218to080320.xlsx"
    
        
     #   print("\n")    
     #   answer=input("Load series?")
     #   if answer=="y":
    
        filename="shopsales010114to070320.xlsx"
    
        print("loading series....",filename) 
    #   series2,product_names=load_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
        series2,product_names=load_shop_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
      #  series2,product_names=load_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
    
        print("Saving series2")
        np.save("series2.npy",series2)
        print("Saving product names")
        np.save("product_names.npy",np.asarray(product_names))

        print("\nProduct names, length=",product_names,len(product_names))
        
        
        
       #  series2=np.array([48,  16,  24,  56,   8,  16,  48,  64,  40,  16,  48,  24,  32,  24,  24,  24,  16,  96,
       #  8,  48,  32,  64,  16,  24,  64,  32,  56,  64,   8,  48,  16,  56,  48,  24,  32, 136,
       #  40,   8,  80,  40,  16,  24,  40,  16,  24,  40,  24,  24,  32,  40,  48,  32,  24,  40,
       #  32,  32,  56,  24,  32,  24,  24,  48,  16,  56,  48,  32,  32,  64,  88,  32,  24,   8,
       #  40,  48,  24,  32, 104,  24,  56,  40,  32, 56]).astype(np.float32).reshape(1,-1,1)
       #  product_names=["test"]
       # # print("series2=\n",series2.shape)
        
    
        print("file loaded shape=",series2.shape)  
       
      
       ###############################################
    #   build model and train using random mini batches of 20 long
     #   print("\n")    
     #   answer=input("Rebuild batches?")
     #   if answer=="y":
        print("Build batches")
       
    #   build_mini_batch_input(series,no_of_batches,no_of_steps)
        X,y=build_mini_batch_input2(series2,no_of_batches,batch_length)
    else:
        print("Load batches")
        X=np.load("batch_train_X.npy")
        y=np.load("batch_train_y.npy")
        series2=np.load("series2.npy")     
        product_names=list(np.load("product_names.npy"))
    
    
    print("\n")  
    n_query_rows=X.shape[2]
    n_steps=X.shape[1]-1
    n_inputs=X.shape[2]
    max_y=np.max(X)
        
    x_axis=np.arange(0,n_steps+1+predict_ahead_steps,1)
        
    #   series=create_extra_batches2(series2,extra_batches,no_of_batch_copies)     # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point.  and then duplicate them 
    for p in range(0,series2.shape[0]):
        plt.figure(figsize=(11,4))
        plt.subplot(121)
        plt.title("A unit sales series: "+str(product_names[p]),fontsize=14)
    
       #plt.title("A unit sales series", fontsize=14)
    
        plt.plot(range(0,series2.shape[1]), series2[p,:], "b-",label=r"$Units$")
      #  plt.plot(x_axis, series2[0,:], "b-",label=r"$unit sales$")
    
    #   plt.plot(x_axis[:-1], sales[product_row_no,x_axis[:-1]], "b-", linewidth=3, label="A training instance")
        plt.legend(loc="best", fontsize=14)
        plt.axis([0, series2.shape[1]+predict_ahead_steps+1, 0, np.max(series2)+10])
        plt.xlabel("Period")
        plt.ylabel("dollars")
       
        plt.show()


    print("epochs=",epochs)
    print("n_query_rows=",n_query_rows)    
    print("n_steps=",n_steps)
    print("n_inputs=",n_inputs)
    print("predict_ahead_steps=",predict_ahead_steps)
   

    print("max y=",max_y)



 #   print("mini_batches X shape=",X[0],X.shape)  
 #   print("mini_batches y shape=",y[0],y.shape)  
   
    batch_size=X.shape[0]
    print("Batch size=",batch_size)

    np.save("batch_train_X.npy",X)
    np.save("batch_train_y.npy",y)
   
   
    train_size=int(round(batch_size*train_percent,0))
    validate_size=int(round(batch_size*validate_percent,0))
    test_size=int(round(batch_size*test_percent,0))
  
 #  print("train_size=",train_size)
 #  print("validate_size=",validate_size)
 #  print("test_size=",test_size)
   
    X_train, y_train = X[:train_size, :,:], y[:train_size,:,:]
    X_valid, y_valid = X[train_size:train_size+validate_size, :,:], y[train_size:train_size+validate_size,:,:]
    X_test, y_test = X[train_size+validate_size:, :,:], y[train_size+validate_size:,:,:]
   # X_all, y_all = series2,series2[-1]
   
  # print("\npredict series shape",series.shape)
    print("X_train shape, y_train",X_train.shape, y_train.shape)
    print("X_valid shape, y_valid",X_valid.shape, y_valid.shape)
    print("X_test shape, y_test",X_test.shape, y_test.shape)
   
    
    answer=input("Recalculate a new model?")
    if answer=="y":

        
        
        
 ##############################           
    
        # print("\nUsing a SimpleRNN shallow layer model.")
        
        # np.random.seed(42)
        # tf.random.set_seed(42)
        
        # model = keras.models.Sequential([
        #     keras.layers.SimpleRNN(batch_length, input_shape=[None,1]),   #[none,1]
        #     keras.layers.Dense(1)
        # ])
        
        # model.compile(loss="mse", optimizer="adam")
        # history = model.fit(X_train, y_train, epochs=epochs,
        #                     validation_data=(X_valid, y_valid))
        
        # model.summary()
        
        # print("\nsave model\n")
        # model.save("simple_shallow_sales_predict_model.h5")
       
################################
        

    
  #       print("\nUsing a SimpleRNN layer model. inputs/outputs=",n_query_rows)
        
  #       np.random.seed(42)
  #       tf.random.set_seed(42)
        
  #       model = keras.models.Sequential([
  # #          keras.layers.SimpleRNN(batch_length*n_query_rows, return_sequences=True, input_shape=[None,n_query_rows]),   #[none,1]
  #           keras.layers.SimpleRNN(80, return_sequences=True, input_shape=[None,n_query_rows]),   #[none,1]
  #           keras.layers.SimpleRNN(12),

  #   #        keras.layers.SimpleRNN(batch_length),
  #           keras.layers.Dense(n_query_rows)
  #       ])
        
  #       model.compile(loss="mse", optimizer="adam")
  #       history = model.fit(X_train, y_train, epochs=epochs,
  #                           validation_data=(X_valid, y_valid))
        
     
         
        
  #       print("\nsave model\n")
  #       model.save("simple_sales_predict_model.h5")
        
  #       model.summary()

        
 ######################################       
 
        

    
        # print("\nUsing a SimpleRNN layer model.")
        
        # np.random.seed(42)
        # tf.random.set_seed(42)
        
        # model = keras.models.Sequential([
        #     keras.layers.SimpleRNN(batch_length, return_sequences=True, input_shape=[None,1]),   #[none,1]
        #     keras.layers.SimpleRNN(20),
        #     keras.layers.Dense(1)
        # ])
        
        # model.compile(loss="mse", optimizer="adam")
        # history = model.fit(X_train, y_train, epochs=epochs,
        #                     validation_data=(X_valid, y_valid))
        
     
         
        
        # print("\nsave model\n")
        # model.save("simple_salesx2_predict_model.h5")
        
        # model.summary()


####################################
               
        # print("Deep RNN with batch norm")
        
        
        # np.random.seed(42)
        # tf.random.set_seed(42)
        
        # model = keras.models.Sequential([
        #     keras.layers.SimpleRNN(batch_length, return_sequences=True, input_shape=[None, 1]),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.SimpleRNN(20, return_sequences=True),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.TimeDistributed(keras.layers.Dense(1))
        # ])
        
        # model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        # history = model.fit(X_train, y_train, epochs=epochs,
        #                     validation_data=(X_valid, y_valid))


        # print("\nsave model Deep RNN\n")
        # model.save("Deep_RNN_with_batch_norm_sales_predict_model.h5")
         
        
        
        # model.summary()
        
   
##################
     
        
                
        # print("LSTMs")
        
        # np.random.seed(42)
        # tf.random.set_seed(42)
        
        # model = keras.models.Sequential([
        #     keras.layers.LSTM(batch_length, return_sequences=True, input_shape=[None, 1]),
        #     keras.layers.LSTM(20, return_sequences=True),
        #     keras.layers.TimeDistributed(keras.layers.Dense(1))
        # ])
        
        # model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        # history = model.fit(X_train, y_train, epochs=epochs,
        #                     validation_data=(X_valid, y_valid))
  

      
        # print("\nsave model LSTM\n")
        # model.save("LSTM_sales_predict_model.h5")
    
    
        # model.summary()
     
        
     #   model.evaluate(X_valid, y_valid)
        
     #   plot_learning_curves(history.history["loss"], history.history["val_loss"])
     #   plt.show()
        
################################3

        # print("GRUs")
        
        # np.random.seed(42)
        # tf.random.set_seed(42)
        
        # model = keras.models.Sequential([
        #     keras.layers.GRU(batch_length, return_sequences=True, input_shape=[None, n_query_rows]),
        #     keras.layers.GRU(batch_length, return_sequences=True),
        #     keras.layers.TimeDistributed(keras.layers.Dense(n_query_rows))
        # ])
        
        # model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        # history = model.fit(X_train, y_train, epochs=epochs,
        #                     validation_data=(X_valid, y_valid))
                
            
        # print("\nsave model GRU\n")
        # model.save("GRU_sales_predict_model.h5", include_optimizer=True)
    
    
        # model.summary()
   
    
    
        
    ################################################3   

        print("\nCNN + GRU's")
        
        np.random.seed(42)
        tf.random.set_seed(42)

        model = keras.models.Sequential([
            keras.layers.Conv1D(filters=batch_length,kernel_size=4,strides=1, padding='same',input_shape=[None,n_query_rows]),  # padding ='valid'
            keras.layers.GRU(batch_length, return_sequences=True),
            keras.layers.GRU(batch_length, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(n_query_rows))
        ])
        
        model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        history = model.fit(X_train, y_train, epochs=epochs,
                            validation_data=(X_valid, y_valid))
                
            
        print("\nsave model CNN\n")
        model.save("CNN_sales_predict_model.h5", include_optimizer=True)
    
    
        model.summary()
   
    
###########################################################################################
        
        print("\nwavenet")
        
        np.random.seed(42)
        tf.random.set_seed(42)

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[None,n_query_rows]))
        for rate in (1,2,4,8) *2:
            model.add(keras.layers.Conv1D(filters=batch_length, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate))
        model.add(keras.layers.Conv1D(filters=n_query_rows, kernel_size=1))    
        model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        history = model.fit(X_train, y_train, epochs=epochs,
                            validation_data=(X_valid, y_valid))
                
            
        print("\nsave model wavenet\n")
        model.save("wavenet_sales_predict_model.h5", include_optimizer=True)
    
    
        model.summary()
     



















#########################################################3                 
    #  test the model    
    
     #    print("testing the model")
    
    
            
     # #   for product_row_no in range(0,n_rows):
     #        #plt.title("Testing the model", fontsize=14)
     #    y_pred = model.predict(X_valid)[0]   #series2[0,:].reshape(1,-1,1))
     #    print("X_valid y_pred=",y_pred,y_pred.shape)   #[:,-n_steps:,:]
 
     #    for p in range(0,n_query_rows):

     #        plt.title("Testing the model: "+str(product_names[p]),fontsize=14)
     #        plt.plot(x_axis[:n_steps+1], series2[p,:], label=r"$unit sales$")
        
   
     #        plt.plot(x_axis[-predict_ahead_steps], y_pred[p], "r.", markersize=10, label="prediction")
     #  #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        
     
        
     #        plt.legend(loc="best")
     #        plt.xlabel("Week")
        
     #        plt.show()
         
        
     #    print("testing the model finished")


    else:        
        print("\nload model")
 #       model = keras.models.load_model("simple_sales_predict_model.h5")
 #Deep_RNN_with_batch_norm_sales_predict_model.h5
     #   model = keras.models.load_model("Deep_RNN_with_batch_norm_sales_predict_model.h5")
       # model=keras.models.load_model("LSTM_sales_predict_model.h5")
   #     model=keras.models.load_model("GRU_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
        model=keras.models.load_model("CNN_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
      #  model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

     #   model=keras.models.load_model("simple_shallow_sales_predict_model.h5")


#############################################    
    
    print("Validating")
    
 #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    
 #   print("series2 shape=",series2.shape)
    n_steps=X_valid.shape[1]
 #   current_series=np.swapaxes(series2,0,2)  #[np.newaxis,:]
 #   print("swapped current series=\n",current_series[:10],current_series.shape)
    
 #   X = X_new
 #   n_steps=X.shape[1]
 #   print("X_valid shape=",X_valid.shape,"n_steps=",n_steps)
 #   ys= model.predict(X[:,1:])[:, np.newaxis,:]
    #print("cureent series=\n",current_series,current_series.shape)
    ys= model.predict(X_valid)    #[:, np.newaxis,:]
 #   print("ys shape=",ys.shape)
  #  ys=ys[np.newaxis,:]
  #  print("afetr ys shape=",ys.shape)

  #  Y_new=np.concatenate((current_series,ys),axis=1)
  #  print("Y_new",Y_new,Y_new.shape)
 #    for step_ahead in range(1,predict_ahead_steps):
 # # #         print("step ahead=",step_ahead,"X[:,step_ahead:].shape=",X[:,step_ahead:].shape)
                    
 #         y_pred_one = model.predict(Y_new[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
 #         print("y_pred_one shape=",y_pred_one,y_pred_one.shape)
 #         y_pred_one=y_pred_one[:,np.newaxis,:]    
 #         print("after ypred one shape=",y_pred_one.shape)

 # #      #   print("y_pred_one=",y_pred_one,y_pred_one.shape,"ypred=",y_pred_one.reshape(-1))
 #         Y_new = np.concatenate((Y_new, y_pred_one),axis=1)    #[:, np.newaxis,:]), axis=1)
 #  #       print("\rPredict step:",step_ahead,Y_new,"Y_new.shape:",Y_new.shape,flush=True,end="\r")

    
 #  #  print("\n\n")    
 #    print("Y new finished.shape=",Y_new, Y_new.shape)
                      
            
 #   for product_row_no in range(0,n_rows):
        #plt.title("Testing the model", fontsize=14)
    for p in range(0,n_query_rows):

        plt.title("Validate Actual vs Prediction: "+str(product_names[p]),fontsize=14)
        
        #    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
        #    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
        # plt.plot(x_axis[:n_steps], Y_new[0,:n_steps,p], "b-", markersize=5, label="actual")
        # plt.plot(x_axis[n_steps-1:], Y_new[0,n_steps-1:,p], "r-", markersize=5, label="prediction")
        plt.plot(range(0,ys.shape[1]), X_valid[0,:,p], "b*", markersize=8, label="actual")
        plt.plot(range(1,ys.shape[1]+1), ys[0,:,p], "r.", markersize=8, label="prediction")
        
           
      #  plt.plot(x_axis, Y_pred, "g.", markersize=10, label="prediction")
      #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        
     
        
        plt.legend(loc="best")
        plt.xlabel("Period")
        
        plt.show()
        
        
        
############################################################################3
        
   
    print("CNN Prediction",predict_ahead_steps,"steps ahead.")
    
 #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    sales=np.swapaxes(series2,0,2)
 #   print("sales shape=",sales.shape)
    n_steps=sales.shape[1]
 #   current_series=np.swapaxes(series2,0,2)  #[np.newaxis,:]
 #   print("swapped current series=\n",current_series[:10],current_series.shape)
    
 #   X = X_new
 #   n_steps=X.shape[1]
 #   print("sales shape=",sales.shape,"n_steps=",n_steps)
 #   ys= model.predict(X[:,1:])[:, np.newaxis,:]
    #print("cureent series=\n",current_series,current_series.shape)
    ys= model.predict(sales)    #[:, np.newaxis,:]
 #   print("start ys shape=",ys.shape)

 #   steps_ahead=1
  #  ys=ys[np.newaxis,:]
  #  print("afetr ys shape=",ys.shape)

  #  Y_new=np.concatenate((current_series,ys),axis=1)
  #  print("Y_new",Y_new,Y_new.shape)
    for step_ahead in tf.range(1,predict_ahead_steps):
 # # #         print("step ahead=",step_ahead,"X[:,step_ahead:].shape=",X[:,step_ahead:].shape)
                    
          y_pred_one = model.predict(ys[:,:step_ahead,:])[:, tf.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
  #        print("y_pred_one shape=",y_pred_one.shape)
  #        print("y pred last one",y_pred_one[:,:,-1,:],y_pred_one[:,:,-1,:].shape)
 #         y_pred_one=y_pred_one[:,np.newaxis,:]    
 #         print("after ypred one shape=",y_pred_one.shape)

 # #      #   print("y_pred_one=",y_pred_one,y_pred_one.shape,"ypred=",y_pred_one.reshape(-1))
          ys = tf.concat((ys, y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
 #  #       print("\rPredict step:",step_ahead,Y_new,"Y_new.shape:",Y_new.shape,flush=True,end="\r")

    
 #  #  print("\n\n")    
 #   print("ys new finished.shape=",ys, ys.shape)
                      
            
 #   for product_row_no in range(0,n_rows):
        #plt.title("Testing the model", fontsize=14)
    for p in range(0,n_query_rows):

        plt.title("Prediction: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
        
        #    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
        #    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
        # plt.plot(x_axis[:n_steps], Y_new[0,:n_steps,p], "b-", markersize=5, label="actual")
        # plt.plot(x_axis[n_steps-1:], Y_new[0,n_steps-1:,p], "r-", markersize=5, label="prediction")
        plt.plot(range(0,sales.shape[1]), sales[0,:,p], "b-", markersize=5, label="actual")
        plt.plot(range(sales.shape[1]-1,sales.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps+1):,p], "r-", markersize=5, label="prediction")
        plt.plot(range(1,ys.shape[1]+1), ys[0,:,p], "g.", markersize=5, label="validation")
        
           
      #  plt.plot(x_axis, Y_pred, "g.", markersize=10, label="prediction")
      #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        
     
        
        plt.legend(loc="best")
        plt.xlabel("Period")
        
        plt.show()
         
  ############################################################################3
        
   
    print("Wavenet Predicting",predict_ahead_steps,"steps ahead.")
    
    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
   
 #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    sales=np.swapaxes(series2,0,2)
 #   print("sales shape=",sales.shape)
    n_steps=sales.shape[1]
 #   current_series=np.swapaxes(series2,0,2)  #[np.newaxis,:]
 #   print("swapped current series=\n",current_series[:10],current_series.shape)
    
 #   X = X_new
 #   n_steps=X.shape[1]
 #   print("sales shape=",sales.shape,"n_steps=",n_steps)
 #   ys= model.predict(X[:,1:])[:, np.newaxis,:]
    #print("cureent series=\n",current_series,current_series.shape)
    ys= model.predict(sales)    #[:, np.newaxis,:]
 #   print("start ys shape=",ys.shape)

 #   steps_ahead=1
  #  ys=ys[np.newaxis,:]
  #  print("afetr ys shape=",ys.shape)
    pas=tf.constant(predict_ahead_steps)   #,dtype=tf.none)
    
    step_ahead=tf.constant(1,dtype=tf.int32)

    for step_ahead in tf.range(1,pas):
          y_pred_one = model.predict(ys[:,:step_ahead,:])[:, tf.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
          ys = tf.concat((ys, y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)

   


  #  Y_new=np.concatenate((current_series,ys),axis=1)
  #  print("Y_new",Y_new,Y_new.shape)
    # pas=tf.constant(predict_ahead_steps,dtype=float)
    
    # step_ahead=tf.constant(1,dtype=float)
          
 #    print("try with a tf.while_loop")      
 #    def cond_func(ys,i,step_ahead):
 #         return tf.math.less_equal(i,pas)
 
 #    def body_func(ys,i,step_ahead):
 #         y_pred_one = model.predict(ys[:,:i,:])[:, tf.newaxis,:]  
 #         return [tf.concat((ys, y_pred_one[:,:,-1,:]),axis=1),tf.add(i,1),step_ahead]

        

 #    ys=tf.while_loop(cond_func,body_func,[ys,1,step_ahead])
 # #       y_pred_one = model.predict(ys[:,:step_ahead,:])[:, tf.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
 #       ys = tf.concat((ys, y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
   #     tf.add(step_ahead,1))


# def cond(t1, t2, i, iters):
#     return tf.less(i, iters)

# def body(t1, t2, i, iters):
#     return [tf.add(t1, 1), t2, tf.add(i, 1), iters]

# t1 = tf.constant(1)
# t2 = tf.constant(5)
# iters = tf.constant(3)

# res = tf.while_loop(cond, body, [t1, t2, 0, iters])



    
 #  #  print("\n\n")    
  #  ys=ys[0]
  #  print("ys new finished.shape=",ys[0], ys[0].shape)
                      
            
 #   for product_row_no in range(0,n_rows):
        #plt.title("Testing the model", fontsize=14)
    for p in range(0,n_query_rows):

        plt.title("Wavenet Prediction: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
        
        #    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
        #    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
        # plt.plot(x_axis[:n_steps], Y_new[0,:n_steps,p], "b-", markersize=5, label="actual")
        # plt.plot(x_axis[n_steps-1:], Y_new[0,n_steps-1:,p], "r-", markersize=5, label="prediction")
        plt.plot(range(0,sales.shape[1]), sales[0,:,p], "b-", markersize=5, label="actual")
        plt.plot(range(sales.shape[1]-1,sales.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps+1):,p], "r-", markersize=5, label="prediction")
        plt.plot(range(1,ys.shape[1]+1), ys[0,:,p], "g.", markersize=5, label="validation")
        
           
      #  plt.plot(x_axis, Y_pred, "g.", markersize=10, label="prediction")
      #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        
     
        
        plt.legend(loc="best")
        plt.xlabel("Period")
        
        plt.show()
         
  
        
       
    
       
#######################################################################################    
    
 #   print("\nload model")
    #model = keras.models.load_model("simple_sales_predict_model.h5")
   # model = keras.models.load_model("Deep_RNN_with_batch_norm_sales_predict_model.h5")
 #   model=keras.models.load_model("GRU_sales_predict_model.h5")
  #  model=keras.models.load_model("simple_shallow_sales_predict_model.h5")

    
  #  print("validating existing sales using trained model",n_steps,".")
    
  #  X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    
 #   print("Y_new shape=",Y_new.shape)
  #  n_steps=Y_new.shape[1]
 #   current_series=Y_new[:,batch_length:,:].reshape(1,-1,1)  #[np.newaxis,:]
  #  series2=Y_new.reshape(1,-1,1)  #[np.newaxis,:]

#    print("series2=\n",series2.shape)
    
 #    X = np.swapaxes(series2,0,2)
    
 # #   n_steps=X.shape[1]
 # #   print("X_shape=",X,X.shape,"n_steps=",n_steps)
 # #   ys= model.predict(X[:,1:])[:, np.newaxis,:]
 #    X_new= model.predict(X[:,:batch_length,:])[:, np.newaxis,:]
 #  #  X=np.roll(X,-1,axis=1)

 # #   print("ys shape=",ys.shape)
 #  #  X=ys[:, np.newaxis,:]
 # #   print("afetr X ys shape=",X.shape)

 #    #X=ys(np.concatenate((X,ys),axis=1)
 # #   print("X_new",X_new,X_new.shape)
 #    for step_ahead in range(1,n_steps-batch_length):
 #        ys_one= model.predict(X[:,:step_ahead+batch_length,:])[:, np.newaxis,:]
 #        print("step ahead=",step_ahead, ys_one)
 #        X_new = np.concatenate((X_new,ys_one),axis=1)    #[:, np.newaxis,:]), axis=1)
 #     #   X=np.roll(X,-1,axis=1)
 #     #   print("X[:,:5]",X[0,:5])
 #  #  print("\n\n")    
 # #   X_new=X_new[:,-n_steps:,:]    
 #   # print("X_new finished.shape=",X_new, X_new.shape)
   
 #    for p in range(0,n_query_rows):
         
                
 #     #   for product_row_no in range(0,n_rows):
 #            #plt.title("Testing the model", fontsize=14)
 #        plt.title("Validating the model: "+str(product_names[p]),fontsize=14)
 #        plt.axis([0, n_steps+1, 0, max_y+10])
    
 #        #    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
 #        #    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
 #        plt.plot(x_axis[:n_steps], series2[p,:n_steps], "b*", markersize=7, label="actual")
 #        plt.plot(x_axis[batch_length:n_steps], X_new[0,:,p], "r.", markersize=5, label="prediction")
        
           
 #      #  plt.plot(x_axis, Y_pred, "g.", markersize=10, label="prediction")
 #      #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        
     
        
 #        plt.legend(loc="best")
 #        plt.xlabel("Period")
        
 #        plt.show()
   
#######################################    
    
   
    
    
  #   print("Validate model & learning curve")
  #   new_steps=n_steps   #Y_new.shape[1]-1-batch_length
  #   print("X_new shape, Y_new shape",X_new.shape,Y_new.shape)

    
  #   X,Y=build_mini_batch_input2(np.swapaxes(Y_new,0,2),no_of_batches,new_steps)
  #   print("X shape,Y shape, Y_new shape",X.shape,Y.shape)
  #   print("number of steps in series from the start to be validated : n_steps=",n_steps)
  #   #new_steps=X.shape[1]
  
  # #  print("new steps the number of -new_steps: ",new_steps)
    
  #   X_train, Y_train = X[:train_size, :new_steps,:], Y[:train_size,:new_steps,:]
  #   X_valid, Y_valid = X[train_size:train_size+validate_size, :new_steps,:], Y[train_size:train_size+validate_size,:,:]
  #   X_test, Y_test = X[train_size+validate_size:, :new_steps,:], Y[train_size+validate_size:,:new_steps,:]
    

  #  # print("\npredict series shape",series.shape)
  #   print("X_train shape, Y_train",X_train.shape, Y_train.shape)
  #   print("X_valid shape, Y_valid",X_valid.shape, Y_valid.shape)
  #   print("X_test shape, Y_test",X_test.shape, Y_test.shape)

    
    
    # np.random.seed(42)
    # tf.random.set_seed(42)
    
    # model = keras.models.Sequential([
    #     keras.layers.Flatten(input_shape=[n_steps, n_query_rows]),   # 50
    #     keras.layers.Dense(n_query_rows)     # 10
    # ])
    
    # model.compile(loss="mse", optimizer="adam")
    # history = model.fit(X_train, y_train, epochs=epochs,
    #                     validation_data=(X_valid, y_valid))  
    
    
    # model.evaluate(X_valid, y_valid)
    
    # plot_learning_curves("Learning curve:"+str(product_names),epochs,history.history["loss"], history.history["val_loss"])
    # plt.show()
    
    
    ################################################3
    
    print("write predictions to CSV file....")
    
    Y_new=list(np.asarray(ys))
   # #     print(sequence)
    with open("sales_prediction_"+str(product_names[0])+".csv", 'w') as csvfile:
        s_writer=csv.writer(csvfile) 
        s_writer.writerow(product_names)
        s_writer.writerows(Y_new[0])

    
    
    print("\n\nFinished.")
    
    
    return





if __name__ == '__main__':
    main()


    
          
          
          
