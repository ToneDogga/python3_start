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
    df["period"]=df.date.dt.to_period('W')    # business days  'D' is every day

  #  periods=set(list(df['period']))
    dates=list(df['period'].dt.strftime("%Y-%m-%d"))
    dates.sort()
 #   mask = mask.replace('"','').strip()    
 #   print("mask=",mask)
    
 #   mask=((df['code']=='FLPAS') & (df['product']=='SJ300'))

    mask=(df['product']=='SJ300')
  #  mask=(df['code']=='FLPAS')
 #   mask=((df['code']=='FLPAS') & (df['product']=="SJ300") & (df['glset']=="NAT"))
#    mask=((df['productgroup']>=10) & (df['productgroup']<=14))
  
 #   mask=((df['code']=='FLPAS') & ((df['product']=="CAR280") | (df['product']=="SJ300")))
  #  mask=((df['code']=='FLPAS') & (df['productgroup']==10))  # & ((df['product']=='SJ300') | (df['product']=='AJ300')))
 #   mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='AJ300') | (df['product']=='TS300')))

   # print("mask=",str(mask))
    print("pivot table being created.")
  #  table = pd.pivot_table(df[mask], values='qty', index=['code','productgroup','product'],columns=['week'], aggfunc=np.sum, margins=False,observed=False, fill_value=0)   #observed=True
  #  table = pd.pivot_table(df[mask], values='qty', index=['code','product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
  #  table = pd.pivot_table(df[mask], values=['qty'], index=['productgroup'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
  #  table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
    table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
 
    print("\ntable=\n",table)   #.head(5))#  #   f.write("\n\n"+table.to_string())
    print("table created.")
    
    product_names=list(table.index)   #"t"  #list(table[table["product"]])

    mat_table=mat_convert(table,52)
    mat_table_90=mat_convert(table,13)

    mat_sales=mat_table.to_numpy()
    mat_sales_90=mat_table_90.to_numpy()

    #print("product names=",product_names)
#    table.drop(columns=['glset'],axis=0)
    sales=table.to_numpy()
 #   sales=sales[...,:n_steps]
 #   print("sales=\n",sales,sales.shape)        
    return sales[..., np.newaxis].astype(np.float32),product_names,dates,table,mat_table,mat_sales[..., np.newaxis].astype(np.float32),mat_sales_90[..., np.newaxis].astype(np.float32)




def load_shop_data(filename):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows


    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

  #  df["period"]=df.date.dt.to_period('W')
  #  df["period"]=df.date.dt.to_period('B')    # business days  'D' is every day
    #df["period"]=df.date.dt.to_period('W')    # business days  'D' is every day
    df["period"]=df.date.dt.to_period('D')    # business days  'D' is every day

  #  dates=list(set(list(df['period'].dt.strftime("%Y-%m-%d"))))
    dates=list(set(list(df['period'].dt.strftime("%Y-%m-%d"))))
    dates.sort()
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
   # table = pd.pivot_table(df[mask], values=['qty'], index=['code','product'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
 #   table = pd.pivot_table(df, values=['salesval'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False, fill_value=0)   #observed=True
    table = pd.pivot_table(df, values=['salesval'],index=['productgroup'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False, fill_value=0)   #observed=True

    print("\ntable=\n",table)   #.head(5))#  #   f.write("\n\n"+table.to_string())
    print("table created.")
    
    #product_names=[""]  #
    product_names=list(table.index)   #"t"  #list(table[table["product"]])
    
    mat_table=mat_convert(table,365)
    mat_sales=mat_table.to_numpy()
 #   periods=list(table['period'])
  #  print("periods",periods)
    mat_table_90=mat_convert(table,90)
    mat_sales_90=mat_table_90.to_numpy()

 
   #print("product names=",product_names)
#    table.drop(columns=['glset'],axis=0)
    sales=table.to_numpy()
 #   sales=sales[...,:n_steps]
 #   print("sales=\n",sales,sales.shape)        
    return sales[..., np.newaxis].astype(np.float32),product_names,dates,table,mat_table,mat_sales[..., np.newaxis].astype(np.float32),mat_sales_90[..., np.newaxis].astype(np.float32)







def mat_convert(table,window_period):
      rtable= table.T.rolling(window=window_period).mean()
      start_mean=rtable[:window_period].mean()
      print("rtable=",rtable,"rtable.columns",rtable.columns,"rtable shape=",rtable.shape,"start mean=",start_mean)
      return rtable.fillna(start_mean)



# def build_mini_batch_input(series,no_of_batches,no_of_steps):
#     print("build mini batch series shape",series.shape)
#     np.random.seed(42) 
#     series_steps_size=series.shape[1]
#   #  print("series steps size=",series_steps_size)

# #  input series array is structured as a 2d array [[step values]]  shape eg (1,82)
# # the numnber of mini batches should be the same length as the input layer number of neurons
# # the output array is structured as a 3D array [no of mini-batches,steps,step values]
# # each mini batch is a training series of say 20 steps long, with a random starting point somewhere on the batch size.
# # if the starting point cannot go too close to the end of the series as it would
# # overflow over the end.
# # also how does it predict the first 20 steps?
    
    
#     # series shape is (2,91,1) say
#     # that is n_query_rows=2
#     # n_steps=91
#     # n_inputs=1
    
    
#     # I need to change to   series shape (1,91,2)
#     # n_query_rows=1
#     # n_steps=91
#     # n_inputs =2 
#  #   print("series shape before change:",series[0,:10],series.shape)
    
#     series=np.swapaxes(series,0,2)
          
#   #  print("series shape after change:",series[0,:10],series.shape)

#     random_offsets=np.random.randint(0,series_steps_size-no_of_steps,size=(no_of_batches)).tolist()
#   #  print("random_offsets=",random_offsets)   #,random_offset.shape)
#  #   single_batch=series[:,:no_of_steps]
#   #  new_mini_batch=np.roll(series[:,:no_of_steps+1],random_offsets[0],axis=1).astype(int) 
#     new_mini_batch=series[:,random_offsets[0]:random_offsets[0]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)

#     prediction=new_mini_batch[:,-1]
#     for i in range(1,no_of_batches):
#         temp=new_mini_batch[:,:no_of_steps]
#         new_mini_batch=series[:,random_offsets[i]:random_offsets[i]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)
#         prediction=np.concatenate((prediction,new_mini_batch[:,-1]))
#         new_mini_batch=np.vstack((temp,new_mini_batch[:,:no_of_steps]))
        
#         if i%100==0:
#             print("\rBatch:",i,"new_mini_batch.shape:",new_mini_batch.shape,flush=True,end="\r")

        
#  #   print("new_mini_batch=",new_mini_batch,new_mini_batch.shape)
#  #   print("prediction=",prediction)
#     print("\n")        
#     return new_mini_batch[:,:no_of_steps],prediction 





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

   # n_steps = 100
    predict_ahead_steps=80
    epochs_cnn=1
    epochs_wavenet=12
    no_of_batches=40000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=4   #731   #731  #365  3 years of days  1096
    y_length=1
    neurons=120
#    overlap=0  # the over lap between the X series and the target y

    kernel_size=4   # for CNN
    strides=2   # for CNN
    
    
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
        filename="NAT-raw310120all.xlsx"
   #     filename="allsalestrans020218-190320.xlsx"
    
    
     
      #  filename="shopsales020218to070320.xlsx"
    
        print("loading series....",filename) 
    #   series2,product_names,periods=load_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
    #    series2,product_names,ddates,table,mat_table=load_shop_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
        series2,product_names,ddates,table,mat_table,mat_sales,mat_sales_90=load_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
    
        print("Saving series2")
        np.save("series2.npy",series2)
        print("Saving product names")
        np.save("product_names.npy",np.asarray(product_names))
        print("Saving dates",len(ddates))
     #   np.save("periods.npy",np.asarray(dates),allow_pickle=True)

        with open("dates.pkl", 'wb') as f:
            pickle.dump(ddates, f)
 
        print("Saving pivot table")
        pd.to_pickle(table, "table.pkl")

        pd.to_pickle(mat_table, "mat_table.pkl")


        mat_sales=np.swapaxes(mat_sales,0,1)
        print("Saving mat_sales")
        np.save("mat_sales.npy",mat_sales)
 
        mat_sales_90=np.swapaxes(mat_sales_90,0,1)
        print("Saving mat_sales_90")
        np.save("mat_sales_90.npy",mat_sales_90)
 

   #     with open("table.pkl", 'wb') as f:
    #        table.to_pickle(f,compression=None)


        print("\nProduct names, length=",product_names,len(product_names))
               
    
        print("file loaded shape=",series2.shape)  
        print("mat sales.shape=",mat_sales.shape)
      
       ###############################################
    #   build model and train using random mini batches of 20 long
     #   print("\n")    
     #   answer=input("Rebuild batches?")
     #   if answer=="y":
        print("Build batches")
       
    #   build_mini_batch_input(series,no_of_batches,no_of_steps)
       # X,y=build_mini_batch_input2(series2,no_of_batches,batch_length)
        X,y=build_mini_batch_input2(mat_sales_90,no_of_batches,batch_length)

    else:
        print("Load batches")
        X=np.load("batch_train_X.npy")
        y=np.load("batch_train_y.npy")
        series2=np.load("series2.npy")   
        print("loading product_names")
        product_names=list(np.load("product_names.npy"))
       # dates=list(np.load("periods.npy",allow_pickle=True))
        print("loading dates")
        with open('dates.pkl', 'rb') as f:
            ddates = pickle.load(f)   
            
        print("Loading pivot table")        
        table= pd.read_pickle("table.pkl")

        mat_table= pd.read_pickle("mat_table.pkl")
        
        print("Loading mat_sales")
        mat_sales=np.load("mat_sales.npy")

        print("Loading mat_sales_90")
        mat_sales_90=np.load("mat_sales_90.npy")

    #    with open("table.pkl", 'wb') as f:
     #       table=pd.read_pickle(f,compression=None)

    
    
    
    print("\n") 
    print("series2.shape=",series2.shape)
    
    print("mat_sales shape",mat_sales.shape)
    print("mat_sales_90 shape",mat_sales_90.shape)

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
    
        plt.plot(range(0,series2.shape[1]), series2[p,:], "b.",alpha=0.5,label=r"$Units$")
      #  plt.plot(x_axis, series2[0,:], "b-",label=r"$unit sales$")
    
    #   plt.plot(x_axis[:-1], sales[product_row_no,x_axis[:-1]], "b-", linewidth=3, label="A training instance")
        plt.legend(loc="best", fontsize=14)
        plt.axis([0, series2.shape[1]+predict_ahead_steps+1, 0, np.max(series2[p])+10])
        plt.xlabel("Period")
        plt.ylabel("dollars")
       
        plt.show()


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

    np.save("batch_train_X.npy",X)
    np.save("batch_train_y.npy",y)
   
   
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
   
  # print("\npredict series shape",series.shape)
    print("X_train shape, y_train",X_train.shape, y_train.shape)
    print("X_valid shape, y_valid",X_valid.shape, y_valid.shape)
    print("X_test shape, y_test",X_test.shape, y_test.shape)
   
 #########################################################
    
    answer=input("Retrain model(s)?")
    if answer=="y":

    
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

        # print("\nCNN + GRU's")
        
        # np.random.seed(42)
        # tf.random.set_seed(42)

        # model = keras.models.Sequential([
        #     keras.layers.Conv1D(filters=batch_length+1,kernel_size=kernel_size,strides=strides, padding='valid',input_shape=[None,n_query_rows]),  # padding ='valid' or 'same'
        #     keras.layers.GRU(batch_length+1, return_sequences=True),
        #     keras.layers.GRU(batch_length+1, return_sequences=True),
        #     keras.layers.TimeDistributed(keras.layers.Dense(n_query_rows))
        # ])
        
        # model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        # history = model.fit(X_train, y_train[:,(kernel_size-1)::strides], epochs=epochs_cnn,
        #                     validation_data=(X_valid, y_valid[:,(kernel_size-1)::strides]))
                
            
        # print("\nsave model CNN\n")
        # model.save("cnn_sales_predict_model.h5", include_optimizer=True)
    
    
        # model.summary()
   
    
###########################################################################################
        
        print("\nwavenet")
        
        np.random.seed(42)
        tf.random.set_seed(42)

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[None,n_query_rows]))
        for rate in (1,2,4,8) *2:
            model.add(keras.layers.Conv1D(filters=neurons, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate))
        model.add(keras.layers.Conv1D(filters=n_query_rows, kernel_size=1))    
        model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        history = model.fit(X_train, y_train, epochs=epochs_wavenet,
                            validation_data=(X_valid, y_valid))
                
            
        print("\nsave model wavenet\n")
        model.save("wavenet_sales_predict_model.h5", include_optimizer=True)
      
        model.summary()
     

    else:        
        print("\nload model")
 #       model = keras.models.load_model("simple_sales_predict_model.h5")
 #Deep_RNN_with_batch_norm_sales_predict_model.h5
     #   model = keras.models.load_model("Deep_RNN_with_batch_norm_sales_predict_model.h5")
       # model=keras.models.load_model("LSTM_sales_predict_model.h5")
   #     model=keras.models.load_model("GRU_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
        model=keras.models.load_model("cnn_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
      #  model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

     #   model=keras.models.load_model("simple_shallow_sales_predict_model.h5")


#############################################    
    
    # print("Validating the CNN model")
    # model=keras.models.load_model("cnn_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

    # #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    
    # #   print("series2 shape=",series2.shape)
    # n_steps=X_valid.shape[1]
    # ys= model.predict(X_valid)    #[:, np.newaxis,:]
    # for p in range(0,n_query_rows):
    #       plt.title("CNN Validate Actual vs Prediction: "+str(product_names[p]),fontsize=14)
    #       plt.plot(range(0,ys.shape[1]*strides**2+(kernel_size+1),strides), X_valid[0,:,p], "b*", markersize=8, label="actual")
    #       plt.plot(range(1,(ys.shape[1]*strides)+(kernel_size-1)-2,strides), ys[0,:,p], "r.", markersize=8, label="prediction")
    #       plt.legend(loc="best")
    #       plt.xlabel("Period")       
    #       plt.show()
        
#############################################    
    
    print("Validating the wavenet model")
    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

  #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    
  #   print("series2 shape=",series2.shape)
    n_steps=X_valid.shape[1]
    ys= model.predict(X_valid)    #[:, np.newaxis,:]
    for p in range(0,n_query_rows):
          plt.title("Wavenet Validate Actual vs Prediction: "+str(product_names[p]),fontsize=14)
          plt.plot(range(0,ys.shape[1]), X_valid[0,:,p], "b*", markersize=8, label="actual")
          plt.plot(range(1,ys.shape[1]+1), ys[0,:,p], "r.", markersize=4, label="prediction")
          plt.legend(loc="best")
          plt.xlabel("Period")       
          plt.show()
         
         
        
############################################################################3
        
   
    # print("CNN Prediction",predict_ahead_steps,"steps ahead.")
    # model=keras.models.load_model("cnn_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
   
    # #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    # sales=np.swapaxes(series2,0,2)
    # #   print("sales shape=",sales.shape)
    # n_steps=sales.shape[1]
    # ys= model.predict(sales)    #[:, np.newaxis,:]
    # for step_ahead in range(kernel_size,predict_ahead_steps,strides):
    #     y_pred_one = model.predict(ys[:,(kernel_size-1):step_ahead,:])[:, tf.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
    #     ys = np.concatenate((ys, y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)

    # for p in range(0,n_query_rows):
    #     plt.title("CNN Prediction: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
    #     plt.plot(range(0,sales.shape[1]*strides,strides), sales[0,:,p], "b-", markersize=5, label="actual")
    #     plt.plot(range(sales.shape[1]*strides-1,sales.shape[1]+predict_ahead_steps,strides), ys[0,-(predict_ahead_steps+1):,p], "r-", markersize=5, label="prediction")
    #     plt.plot(range(1,ys.shape[1]*strides+1,strides), ys[0,:,p], "g.", markersize=5, label="validation")
        
    #     plt.legend(loc="best")
    #     plt.xlabel("Period")
        
    #     plt.show()
         
  ############################################################################3
        
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors
    
    print("Series Predicting",predict_ahead_steps,"steps ahead.")
    
    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
   
  #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    sales=np.swapaxes(series2,0,2)
    print("sales shape=",sales.shape)
 
  #  mat_swap_sales=np.swapaxes(mat_sales,0,1)
    print("mat__sales shape",mat_sales.shape)
   
    #print("mat sales shape",mat_sales.shape)

    #   n_steps=sales.shape[1]
    #ys= model.predict(sales)    #[:, np.newaxis,:]
 
    ys= model.predict(mat_sales)    #[:, np.newaxis,:]
    pas=predict_ahead_steps   #,dtype=tf.none)
    
 #   step_ahead=tf.constant(1,dtype=tf.int32)

    for step_ahead in range(1,pas):
          print("\rstep:",step_ahead,"/",pas,end='\r',flush=True)
          y_pred_one = model.predict(ys[:,:step_ahead,:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
          ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)

    print("\rstep:",step_ahead+1,"/",pas,end='\n\n',flush=True)

 #   new1_ys=tf.make_ndarray(tf_ys)
  #  new1_ys=np.array(tf_ys)
#################################################################    
    
 #   mat_swap_sales=np.swapaxes(mat_sales,0,1)
 #   print("mat_swap_sales shape",mat_swap_sales.shape)
 #   print("mat sales shape",mat_sales.shape)

    
    for p in range(0,n_query_rows):

        plt.title("Series Prediction: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
        
        plt.plot(range(0,sales.shape[1]), sales[0,:,p], "b-", markersize=5, label="actual wk")
        plt.plot(range(0,mat_sales.shape[1]), mat_sales[0,:,p], "m-", markersize=5, label="actual mat")
        plt.plot(range(0,mat_sales_90.shape[1]), mat_sales_90[0,:,p], "y-", markersize=5, label="actual 90 day MT")

        plt.plot(range(sales.shape[1]-1,sales.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps+1):,p], "r-", markersize=5, label="prediction")
        plt.plot(range(1,ys.shape[1]+1), ys[0,:,p], "g.", markersize=5, label="validation")
       
        plt.legend(loc="best")
        plt.xlabel("Period")
        
        plt.show()
         
  
#####################################################
# wavenet predictions using a tf.while_loop to avoid traceback
#          

 #    print("Wavenet predictions using tf.while_loop")  

 #    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
   
 #    sales=np.swapaxes(series2,0,2)

 #    ys= model.predict(sales)    #[:, np.newaxis,:]
 #    tf_pas=tf.constant(20)  #predict_ahead_steps)   #,dtype=tf.none)
 #    counter=tf.constant(1)   
 #    tf_ys = tf.convert_to_tensor(ys, dtype=tf.float32)


 #    print("tf_ys completed")
 # #   step_ahead=tf.constant(1,dtype=tf.int32)
                 
 #    def cond(tf_ys, counter, tf_pas):
 #        return tf.less_equal(counter,tf_pas)
    
 #    def body(tf_ys, counter, tf_pas):
 #        y_pred_one = model.predict(tf_ys[:,:counter,:])[:, tf.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
 #        tf_new_pred=tf.convert_to_tensor(y_pred_one[:,:,-1,:])
 #        tf_ys = tf.concat((tf_ys,tf_new_pred),axis=1)    #[:, np.newaxis,:]), axis=1)       
 #        return [tf_ys, tf.add(counter, 1), tf_pas]
    
 # #   step_ahead = tf.constant(1)
 #  #  t2 = tf.constant(5)
 #  #  tf_pas = tf.constant(predict_ahead_steps)
    
 #    res=tf.while_loop(cond, body, [tf_ys, counter, tf_pas],maximum_iterations=100)

 #    print("tf_ys=",res[0])   #,"step_ahead=",step_ahead,"tf_ys=",tf_ys,"pas=",pas)
        
 #    new2_ys=tf.make_ndarray(res[0])
    
#    for step_ahead in tf.range(1,pas):
#        y_pred_one = model.predict(ys[:,:step_ahead,:])[:, tf.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
#        ys = tf.concat((ys, y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)

# #################################################################    
    
    # for p in range(0,n_query_rows):

    #     plt.title("Wavenet while_loop Prediction: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
        
    #     plt.plot(range(0,sales.shape[1]), sales[0,:,p], "b-", markersize=5, label="actual")
    #     plt.plot(range(sales.shape[1]-1,sales.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps+1):,p], "r-", markersize=5, label="prediction")
    #     plt.plot(range(1,ys.shape[1]+1), new2_ys[0,:,p], "g.", markersize=5, label="validation")
       
    #     plt.legend(loc="best")
    #     plt.xlabel("Period")
        
    #     plt.show()
 

       
#######################################################################################    
    
    print("\nwrite predictions to CSV file....")
#    dates.sort()
    print("table shape=",table.shape)
    print(table)
    print("ys",ys.shape)    
    
    d=len(ddates)
    
    swap_ys=np.swapaxes(ys,0,2)[:,-predict_ahead_steps:,0]
    
    print("swap_ys 1",swap_ys.shape)    
   # swap_ys=swap_ys[:,:,0]
   # print("swap_ys 2",swap_ys.shape)    
   
    pred_df1=pd.DataFrame(data=swap_ys,   #,    # values
             index=np.array(product_names))  #,   # 1st column as index
            # columns=ddates)  # 1st row as the column names
    
    print("pd=",pred_df1)
  #  print("y=",y)
    print("First date=",ddates[0])
    print("last date=",ddates[-1])
  #  ldate=dt.strptime(ddates[-1])    
  #  print("ldate",ldate)
  #  print("ldate+1",ldate+1)
 #       df["period"]=df.date.dt.to_period('D')    # business days  'D' is every day

 #   periods=set(list(df['period']))
 
    #Y_new=np.asarray(ys[0])
   # #     print(sequence)
    with open("sales_prediction_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
     #   s_writer=csv.writer(csvfile) 
   #     s_writer.writerow(periods)
   #     s_writer.writerow(product_names)
        table.T.to_csv(f)  #,line_terminator='rn')
        print("table.T.shape=",table.T.shape)
        pred_df1.T.to_csv(f)  #,line_terminator='rn')
        print("pred_df1.T.shape=",pred_df1.T.shape)


    #     j=0
    #     for row in product_names:
    # #        s_writer.writerow([row])
    #         day_count=1
    #         for i in range(table.shape[0],ys.shape[1]):
    #          #   print("i=",i,"d=",d)
    #         #    if i>=d:
    #             line=str(row)+", pred day:"+str(day_count)+","+str(ys[0,i,j])+"\n"                  
    #          #   else:    
    #           #      line=str(row)+","+str(ddates[i])+","+str(ys[0,i,j])+"\n"
    #      #       print("line=",line)
    #             f.write(line)
    #             day_count+=1
                
    #         f.write("\n")    
    #         j+=1    
    #  #   s_writer.writerows(dates) 
    
    print("\n\nFinished.")
    
    
    return





if __name__ == '__main__':
    main()


    
          
          
          
