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
#physical_devices = tf.config.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.set_memory_growth(physical_devices[0], True)
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
import matplotlib.cm as cm

print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# # Where to save the figures
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "rnn"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)

# #n_steps = 50

# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)
    
    




    
# def load_data(mats,filename):    #,col_name_list,window_size):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
#     df=pd.read_excel(filename,-1)  # -1 means all rows

#     # ,dtype={'productgroup': int}

#    # df=df.rename(index=int).index
#     #  create a pivot table of code, product, day delta and predicted qty and export back to excel
    
#     df.fillna(0,inplace=True)
    
#   #  print(df['productgroup'])
#     df['productgroup']=(df['productgroup']).astype(int)
#   #  print(df)

    
#  #   df["period"]=df.date.dt.to_period('W')
   
#     df["period"]=df.date.dt.to_period('D')
#        #     dates=list(set(list(series_table.T['period'])))   #.dt.strftime("%Y-%m-%d"))))
#   #  dates=list(set(list(df['period'].dt.strftime("%Y-%m-%d"))))   #.dt.strftime("%Y-%m-%d"))))

#   #  dates.sort()

#   #  df["period"]=df.date.dt.to_period('B')    # business days  'D' is every day
#   #  df["period"]=df.date.dt.to_period('D')    # business days  'D' is every day
#    #  periods=set(list(df['period']))
#   #  dates=list(df['period'].dt.strftime("%Y-%m-%d"))
#   #  dates.sort()
#  #   mask = mask.replace('"','').strip()    
#  #   print("mask=",mask)
    
#  #   mask=((df['code']=='FLPAS') & (df['product']=='SJ300'))

#  #   mask=(df['product']=='SJ300')
#   #  mask=(df['code']=='FLPAS')
#  #   mask=((df['code']=='FLPAS') & (df['product']=="SJ300") & (df['glset']=="NAT"))
# #    df['productgroup'] = df['productgroup'].astype('category')
#  #   mask=((df['productgroup']>=10) & (df['productgroup']<=14))
#     mask=ic.mask
#   #  mask=(df['productgroup']==10)
#  #   mask=((df['code']=='FLPAS') & (df['product']=="SJ300"))
  
#  #   mask=((df['code']=='FLPAS') & ((df['product']=="CAR280") | (df['product']=="SJ300")))
#   #  mask=((df['code']=='FLPAS') & (df['productgroup']==10))  # & ((df['product']=='SJ300') | (df['product']=='AJ300')))
#  #   mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='AJ300') | (df['product']=='TS300')))

#    # print("mask=",str(mask))
#     print("pivot table being created.")
#   #  table = pd.pivot_table(df[mask], values='qty', index=['code','productgroup','product'],columns=['week'], aggfunc=np.sum, margins=False,observed=False, fill_value=0)   #observed=True
#   #  table = pd.pivot_table(df[mask], values='qty', index=['code','product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
#   #  table = pd.pivot_table(df[mask], values=['qty'], index=['productgroup'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
#   #  table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
    
#     dates=list(set(list(df['date'].dt.strftime("%Y-%m-%d"))))
#   #  dates=list(set(list(df['date'].dt.strftime("%Y-%W"))))

#     dates.sort()
#   #  df['date_of_birth'] = pd.to_datetime(df['date_of_birth'],infer_datetime_format=True)
#  #   print("1dates",dates,len(dates))
    
#     df['period'] = df['period'].astype('category')
#    # df[mask]=df[mask].astype('category')

#     table = pd.pivot_table(df[mask], values='qty', index=['productgroup'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True
#   #  table=table.rename(index=str).index
#    # table=table.T
# #    print("\ntable=\n",table)   #.head(5))#  #   f.write("\n\n"+table.to_string())
#  #   print("1table created shape.\n",table,table.shape)
#   #  dates=list(table.index.categories.dt.strftime("%Y-%m-%d"))
#  #   print("2dates=",dates)
#   #  print("2table created shape.\n",table.T,table.T.shape)
    
   
#  #   dates=list(set(list(table['period'].dt.strftime("%Y-%m-%d"))))   #.dt.strftime("%Y-%m-%d"))))
#   #  dates=list(set(list(table.categories.dt.strftime("%Y-%m-%d"))))   #.dt.strftime("%Y-%m-%d"))))

#   #  dates.sort()
    
#   #  print("dates=",dates,len(dates))
    
#    # table=table.T
#    # table['newdate']=dates
#   #  table.insert(0, 'new_date', dates)
#   #  table['new_date'] = pd.to_datetime(table['new_date'],infer_datetime_format=True)
#    # table=table.T
#   #  print("3table created shape.\n",table,table.shape)
    
#   #  colnames=list(table.columns.astype(int).astype(str))
#     colnames=list(table.columns)
#  #   del colnames[0]
#   #  print("cn=",colnames)
#    # table['window']=int(1)
#     for window_length in range(0,len(mats)):
#         col_no=0
#         for col in colnames:
#    #         print("col=",col)
#             table.rename(columns={col: str(col)+"@1"},inplace=True)
#             table=add_mat(table,col_no,col,mats[window_length])
#             col_no+=1
#    #     table['window'].fillna(mats[window_length])

#  #   rtable=table
#  #   print("2table=\n",table,"\n\ntable shape",table.shape)   #,rtable.iloc[:window_period,0],"rtable.columns",rtable.columns,"rtable shape=",rtable.shape,"start mean=",start_mean)
#  #   print("3table=\n",table.T,"\n\ntable shape",table.T.shape)   #,rtable.iloc[:window_period,0],"rtable.columns",rtable.columns,"rtable shape=",rtable.shape,"start mean=",start_mean)

#   #  table=table.T
  

#     table = table.reindex(natsorted(table.columns), axis=1)

  
#    # print("\ntable=\n",table)   #.head(5))#  #   f.write("\n\n"+table.to_string())
#    # print("table created shape.\n",table.shape)
  
#     splittable=remove_some_table_columns(table,mats).T   #,col_name_list,window_size) 
#  #   mat_sales=np.swapaxes(splittable.to_numpy(),0,2)
#  #   print("\n1splittable=\n",splittable)   #.head(5))#  #   f.write("\n\n"+table.to_string())
#  #   print("1splittable created shape.\n",splittable.shape)

#   #  splittable=table.T
    
#  #   print("\n2splittable=\n",splittable)   #.head(5))#  #   f.write("\n\n"+table.to_string())
#  #   print("2splittable created shape.\n",splittable.shape)


#     # mat_sales=splittable.to_numpy()

#     # print("mat_sales=\n",mat_sales.shape,mat_sales)
 
#     return splittable,dates

# #    return mat_sales[..., np.newaxis].astype(np.float32),dates,splittable   #,product_names,dates,table,mat_table,mat_sales[..., np.newaxis].astype(np.float32),mat_sales[..., np.newaxis].astype(np.float32)




# def add_mat(table,col_no,col_name,window_period):
#    #   print("table iloc[:window period]",table.iloc[:,:window_period])     #shape[1])
#   #  start_mean=table.iloc[:window_period,0].mean(axis=0) 

#     start_mean=table.iloc[:window_period,col_no].mean(axis=0) 
    
# #  print("start mean=",window_period,start_mean)   # axis =1
#     mat_label=str(col_name)+"@"+str(window_period)  
#     table[mat_label]= table.iloc[:,col_no].rolling(window=window_period,axis=0).mean()

#  #   table=table.iloc[:,0].fillna(start_mean)
#     return table.fillna(start_mean)



# def remove_some_table_columns(table,window_size):   #,col_name_list,window_size):
#     # col name is a str
#     # returns anumpy array of the 
#     col_filter="(@"+str(window_size)+")"
#    # for col_name in col_name_list:
#    #     col_filter=col_filter+col_name[:2]+"@|"
#    # col_filter=col_filter+"@"+str(window_size)+")"
#     print("col filter=\n",col_filter)
#     return table.filter(regex=col_filter)

    



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
    print("1cols=",cols)
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
  #  print("series table",series_table)
 #   series_table['period'].remove_categories()
    
    plus_five_years = date.today().year+5
 #   print("+5y=",plus_five_years)
  
  #  transformed_series_table=series_table.T
  #  print("tsc=",transformed_series_table.columns)
    last_date=dates[-1]
    new_dates1=pd.bdate_range(start=last_date, end='3/24/'+str(plus_five_years))  # usa format
    #print("1ex test=\n",test)
    
    new_dates2=new_dates1.strftime('%Y-%m-%d').to_list()
 #   print("new dates2=",new_dates2)
    #del test2[0]
  
    extended_series=pd.DataFrame(new_dates2[1:predict_ahead_steps+1],columns=['period'])

 #   extended_series['period'] = extended_series['period'].astype('category')
  #  extended_series=extended_series.T
    #   new_dates=pd.DataFrame([new_dates2[1:predict_ahead_steps+1],0,0],columns=transformed_series_table.columns)
  #  print("1extended series=\n",extended_series,extended_series.columns,extended_series.shape)
    
    for col in series_table.columns:
        if col=='period':
            pass
        else:
            extended_series[col]=0.0
    
  #  extended_series=extended_series.rename(columns={'0':'period'})
    extended_series['period'] = extended_series['period'].astype('category')

 #   print("2extended series=\n",extended_series,extended_series.columns,extended_series.shape)
 
 #   print("2series table=\n",series_table,series_table.columns,series_table.shape)
   
    extended_series2=series_table.append(extended_series)   #,ignore_index=True)  #,right_index=True, left_on='period')
   # extended_series=extended_series.T
    #test3=test2.to_list()
    #print("3ex test3=\n",test3)
    extended_series2.set_index('period', inplace=True)
 #   print("3extended series2=\n",extended_series2,extended_series2.columns,"\n\n index=",extended_series2.index,extended_series2.shape)
 #   extended_series2=extended_series2['period'].index
 #   extended_series2['period'] = extended_series2['period'].astype('category')
 
  #  print("4extended series2=\n",extended_series2,extended_series2.columns,extended_series2.shape)
  
    extended_table3=extended_series2.T
 #   transformed_series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
 #   extended_table=transformed_series_table.append(new_dates,columns=['period'],ignore_index=True)
  #  print("extended_table3=\n",extended_table3,extended_table3.columns,extended_table3.shape)    

  #  extended_dates=extended_series2.index.dt.strftime("%Y-%m-%d")
   #   extended_dates.append(dates,extended_series2.index.tolist())
 #dates=list(set(list(df['date'].dt.strftime("%Y-%m-%d"))))
    exdates=extended_series2.index.astype(str).tolist()  #.astype(str)) #strftime("%Y-%m-%d"))
  #  print("exdates=",exdates)
  #  dates.append(exdates)   #.astype(str)) #strftime("%Y-%m-%d"))
 #   extended_dates=list(set(dates))  #.dt.strftime("%Y-%m-%d")))
 #   extended_dates.sort()
 #   print("ed=\n",exdates,len(exdates))
    extended_series=extended_table3.T
  #  print("et=\n",extended_series,extended_series.shape)
  #  extended_dates=extended_series2['period'].to_list()
    return extended_table3,exdates
 
    

def graph_a_series(series_table,dates,column_number): 
    series_table=series_table.T  
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
    ax = plt.gca()
    cols=list(series_table.columns)
    del cols[-1]  # delete reference to period column
   # c=cols[column_number]
  #  col=cols[column_number]
   # print("1c=ols",cols)
  #  print("c=",c)
  #  print("graph a series column_number=",column_number)  
    
    col_count=0
    for col in cols:
        if col_count==column_number:
            series_table.plot(kind='line',x='period',y=col,color="blue",ax=ax,fontsize=8)
        col_count+=1    

   # del cols[-1]  # delete reference to period column
  #  print("2cols=",cols)
   
  #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
    #colors = cm.rainbow(np.linspace(0, 2, len(cols)))

    #for col in cols[column_number]:
      #  color=np.random.rand(len(cols),3)
   # series_table.plot(kind='line',x='period',y=c,color='blue',ax=ax,fontsize=8)

  #  plt.show()
  #  print("graph finished")
    return 
    







   

def graph_a_prediction(series_table,dates,column_number): 
    series_table=series_table.T  
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
    ax = plt.gca()
    cols=list(series_table.columns)
    del cols[-1]  # delete reference to period column
   # c=cols[column_number]
  #  col=cols[column_number]
    print("1c=ols",cols)
    print("seres table=\n",series_table,series_table.shape)
  #  print("c=",c)
  #  print("graph a series column_number=",column_number)  
    
    
    
        
    series_table.plot(kind='line',x='period',y=cols[0],color="blue",ax=ax,fontsize=8)
        
    series_table.plot(kind='line',x='period',y=cols[1],color="green",ax=ax,fontsize=8)
       

   # del cols[-1]  # delete reference to period column
  #  print("2cols=",cols)
   
  #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
    #colors = cm.rainbow(np.linspace(0, 2, len(cols)))

    #for col in cols[column_number]:
      #  color=np.random.rand(len(cols),3)
   # series_table.plot(kind='line',x='period',y=c,color='blue',ax=ax,fontsize=8)

  #  plt.show()
  #  print("graph finished")
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



    
  # def on_train_batch_begin(self, batch, logs=None):
  #   print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  # def on_train_batch_end(self, batch, logs=None):
  #   print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  # def on_test_batch_begin(self, batch, logs=None):
  #   print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  # def on_test_batch_end(self, batch, logs=None):
  #   print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


# class MCDropout(keras.layers.Dropout):
#     def call(self,inputs):
#         return super().call(inputs,training=True)


class MCDropout(keras.layers.AlphaDropout):
    def call(self,inputs):
        return super().call(inputs,training=True)



# def save_model(model,filename):
#     #filename = 'finalized_model.sav'
#     #    joblib.dump(regressor,open("SGDRegressorNS.p","wb"))

#     joblib.dump(model, filename)
#     return 



# def load_model(filename):
#     # some time later...

#     # load the model from disk

#     loaded_model = joblib.load(filename)
#     return loaded_model




def main():

   # n_steps = 100
#     predict_ahead_steps=120
#     epochs_cnn=1
#     epochs_wavenet=40
#     no_of_batches=60000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
#     batch_length=10  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
#     y_length=1
#     neurons=800
#     start_point=20
#  #   plot_y_extra=1000   # extra y to graph plot 
#  #   mat_length=20
# #    overlap=0  # the over lap between the X series and the target y

#     kernel_size=4   # for CNN
#     strides=2   # for CNN
    
    
# # train validate test split 
#     train_percent=0.7
#     validate_percent=0.2
#     test_percent=0.1
    
#####################################################################

    predict_ahead_steps=ic.predict_ahead_steps   # 120
    epochs_cnn=ic.epochs_cnn   # 1
    epochs_wavenet=ic.epochs_wavenet   # 40
    no_of_batches=ic.no_of_batches  #60000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=ic.batch_length   #10  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
    y_length=ic.y_length    #1
    neurons=ic.neurons     #800
    start_point=ic.start_point   #  20
#    predict_start_point=ic.predict_start_point   # 260
 #   plot_y_extra=1000   # extra y to graph plot 
 #   mat_length=20
#    overlap=0  # the over lap between the X series and the target y

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
        series_table,dates=ic.load_data(mats,filename)    #,col_name_list,window_size)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)



        
        product_names=list(series_table.index) 
        print("\nProduct names, length=",product_names,len(product_names))

    #    print("1series table=\n",series_table.T,series_table.T.shape)
   # #     dates=list(set(list(series_table.T['period'])))   #.dt.strftime("%Y-%m-%d"))))
   #      dates=list(set(list(series_table.T.index.strftime("%Y-%m-%d"))))   #.dt.strftime("%Y-%m-%d"))))

   #      dates.sort()
   
     #   print("dates=",dates,len(dates))

###########################################
#  test scal the series table
      #   scaler = StandardScaler()
      # #  scaler= MinMaxScaler()

      #   pre_scaled_series=series_table.to_numpy()
        
      #   print("Saving pre scaled series - pre_scaled_series.npy")
      #   np.save("pre_scaled_series.npy",pre_scaled_series)
         
        
      #   print("\npre scaled_series=\n",pre_scaled_series,pre_scaled_series.shape)

      #   scaled_series=scaler.fit_transform(pre_scaled_series)   #, copy=None)

      #   print("\npost scaled_series=\n",scaled_series,scaled_series.shape)
      #   #print("\nX_scaled=\n",X_scaled,X_scaled.shape)
      # #  X_scaled=X_scaled.reshape(-1,1)
      # #  y_scaled=y_scaled.reshape(-1,1)
      # #  print("\nX_scaled+new axis=\n",X_scaled,X_scaled.shape)
      #     # save the scaler
          
      #   print("Saving the scaled series - scaled_series.npy")
      #   np.save("scaled_series.npy",scaled_series)
  
      #   print("saving scaler")
      #   dump(scaler, open('scaler.pkl', 'wb'))
        
      #   ############
        
      #   print("\nloading scaler")
      #   # load the scaler
      #   scaler = load(open('scaler.pkl', 'rb'))
        
      #   print("Loading the scaled series - scaled_series.npy")
      #   scaled_series=np.load("scaled_series.npy")

      #   inverse_scaled=scaler.inverse_transform(scaled_series)
      #   print("\ninverse scaled_series=\n",inverse_scaled,inverse_scaled.shape)
       
      #   print("\nscaled series loaded",scaled_series,scaled_series.shape)

      #   print("loading pre scaled series - pre_scaled_series.npy")
      #   loaded_pre_scaled_series=np.load("pre_scaled_series.npy")

      #   print("\npre scaled series loaded",loaded_pre_scaled_series,loaded_pre_scaled_series.shape)

      #  X_inverse_scaled=X_scaled[np.newaxis]
     #   print("\nX inverse scaled [0,0]=\n",X_inverse_scaled,X_inverse_scaled.shape)

#############################################      




   
        
        #  this adds 
        graph_whole_pivot_table(series_table,dates)
        
        extended_series_table,extended_dates=extend_pivot_table(series_table,dates,predict_ahead_steps)
        
        graph_whole_pivot_table(extended_series_table,extended_dates)
        
        
        #print("Saving series table with date \n",series_table_with_date)
        #pd.to_pickle(series_table_with_date,"series_table_with_date.pkl")
     
    #    print("Saving pickled table - series_table.pkl")
    #    pd.to_pickle(series_table,"series_table.pkl")
     #    np.save("series2.npy",series2)
     
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
 
     #   np.save("dates.pkl",np.asarray(dates),allow_pickle=True)

   #     mat_sales_x=convert_pivot_table_to_numpy_series(series_table)   #,row_no_list)
        #mat_sales_x=convert_pivot_table_to_numpy_series(scaled_series)   #,row_no_list)
       # mat_sales_x=np.swapaxes(scaled_series,0,1)
        scaled_series=series_table.to_numpy()
   #     scaled_series=series_table.to_numpy()
      
        mat_sales_x=np.swapaxes(scaled_series,0,1)
       # mat_sales_x=np.swapaxes(scaled_series,0,1)
 
        mat_sales_x=mat_sales_x[np.newaxis] 
 
        print("Build batches")
        print("mat sales_x.shape=",mat_sales_x.shape)

       
    #   build_mini_batch_input(series,no_of_batches,no_of_steps)
       # X,y=build_mini_batch_input2(series2,no_of_batches,batch_length)
        X,y=build_mini_batch_input(mat_sales_x,no_of_batches,batch_length)


##############################################
#####################33


# scaler = StandardScaler()
# num_instances, num_time_steps, num_features = train_data.shape
# train_data = np.reshape(train_data, shape=(-1, num_features))
# train_data = scaler.fit_transform(train_data)
# train_data = np.reshape(train_data, shape=(num_instances, num_time_steps, num_features))
# num_instances, num_time_steps, num_features = val_data.shape
# val_data = np.reshape(val_data, shape=(-1, num_features))
# val_data = scaler.transform(val_data)
# val_data = np.reshape(val_data, shape=(num_instances, num_time_steps, num_features))

##########################################

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

 #       print("Loading pivot table with date")        
 #       series_table_with_date= pd.read_pickle("series_table_with_date.pkl")

        print("Loading mat_sales_x")
        mat_sales_x=np.load("mat_sales_x.npy")

    #    print("Loading product names")
    #    product_names=list(np.load("product_names.npy"))

   
           
    
    print("\n\nseries table loaded shape=",series_table.shape,"\n")  
    print("product names=\n",product_names)
  #  print("series table with date=\n",series_table_with_date)
    print("dates array=\n",len(dates))
    
    n_query_rows=X.shape[2]
    n_steps=X.shape[1]-1
    n_inputs=X.shape[2]
    max_y=np.max(X)
      
  #  print("st=\n",series_table)
  
  
 #    graph_count=series_table.shape[0]  
 #    series_count=series_table.shape[1]
    
 # #   series_table=series_table.T  
 #    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)

 # #   cols=list(series_table.columns)
 #  #  
 # #   del cols[-1]  # delete reference to period column
  #  print("1cols=",cols)
  
    
  #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
    #colors = cm.rainbow(np.linspace(0, 2, len(cols)))



  
    
    for p in range(0,series_table.shape[0]):
        plt.figure(figsize=(11,4))
        plt.subplot(121)
        plt.title("A unit sales series: "+str(product_names[p]),fontsize=14)
      
        plt.ylabel("Units")
        plt.xlabel("Period") 
        graph_a_series(series_table,dates,p)
        
        plt.legend(loc="best")
        plt.show()
        
  #   series_table=series_table.T  
  #   series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
  #   ax = plt.gca()
  #   cols=list(series_table.columns)
  # #  print("1cols=",cols)
  #   del cols[-1]  # delete reference to period column
  # #  print("2cols=",cols)
   
  # #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
  #   #colors = cm.rainbow(np.linspace(0, 2, len(cols)))

  #   for col in cols:
  #       color=np.random.rand(len(cols),3)
  #       series_table.plot(kind='line',x='period',y=col,color=color,ax=ax,fontsize=8)

  #   plt.legend(loc="best")
  #   plt.title("Unit sales per day")
  #   plt.ylabel("Units")
  #   plt.xlabel("Period") 
  #   plt.show()
  # #  print("graph finished")
  #   return 
    

  #      print("p=",p)
     
    
         
  #  print("2cols=",cols)
   
  #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
    #colors = cm.rainbow(np.linspace(0, 2, len(cols)))

     #   for col in cols:
    #    color=np.random.rand(len(cols),3)
   #     series_table.plot(kind='line',x='period',y=col,color=color,ax=ax,fontsize=8)

    
   #      ax = plt.gca()
   #      series_table.plot(kind='line',x='period',y=0,color='blue',ax=ax,fontsize=8)
   # #     series_table_with_date.axis(x='period')   #,y=0,color='blue',ax=ax,fontsize=8)


    
   #     #plt.title("A unit sales series", fontsize=14)
    
   #      plt.plot(range(0,series_count), series_table_with_date.iloc[:,p], "b-",alpha=1,label=r"$Units$")
   #   #   plt.plot(series_table.index, series_table.iloc[p,:], "b-",alpha=1,label=r"$Units$")

   #      #  plt.plot(x_axis, series2[0,:], "b-",label=r"$unit sales$")
    
   #  #   plt.plot(x_axis[:-1], sales[product_row_no,x_axis[:-1]], "b-", linewidth=3, label="A training instance")
   #      plt.legend(loc="best", fontsize=14)
   #   #   plt.axis([dates[0], dates[series_table.shape[1]], 0, series_table.iloc[p].max()*1.1])
 
   #      plt.axis([0, series_count,predict_ahead_steps+1, 0, series_table_with_date.iloc[:,p].max()*1.1])
 
   #      plt.xlabel("Period")
   #      plt.ylabel("units")
       
   #      plt.show()



 #   series_table=series_table.T

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

        print("plot learning curve")
        plot_learning_curves("learning curve",epochs_wavenet,history.history["loss"], history.history["val_loss"])
        plt.show()
 
        print("plot log learning curve")       
        plot_log_learning_curves("Log learning curve",epochs_wavenet,history.history["loss"], history.history["val_loss"])
        plt.show()

###################################
        #   measure the models accuracy at different dropout levels
        # print("Test accuracy of predictions")
        # y_values=np.stack([model(X_test,training=True) for sample in range(20)])
        # y_value_mean=y_values.mean(axis=0)
        # y_value_std=y_values.std(axis=0)


        # print("y_value_mean, std=\n",y_value_mean,y_value_std) 
        # print("y_values.shape=",y_values.shape)
        # print("y value mean shape=",y_value_mean.shape)
        # print("y value stddev shape=",y_value_std.shape)
        
        
       
        
        
#############################################3
    else:        
        print("\nload model")
 #       model = keras.models.load_model("simple_sales_predict_model.h5")
 #Deep_RNN_with_batch_norm_sales_predict_model.h5
     #   model = keras.models.load_model("Deep_RNN_with_batch_norm_sales_predict_model.h5")
       # model=keras.models.load_model("LSTM_sales_predict_model.h5")
   #     model=keras.models.load_model("GRU_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
       # model=keras.models.load_model("cnn_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
    #    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

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
    
  #   print("Validating the wavenet model")
  #   model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

  # #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    
  # #   print("series2 shape=",series2.shape)
  #   n_steps=X_valid.shape[1]
  #   ys= model.predict(X_valid)    #[:, np.newaxis,:]
  #   for p in range(0,n_query_rows):
  #         plt.title("Wavenet Validate Actual vs Prediction: "+str(product_names[p]),fontsize=14)
  #         plt.plot(range(0,ys.shape[1]), X_valid[0,:,p], "b*", markersize=8, label="actual")
  #         plt.plot(range(1,ys.shape[1]+1), ys[0,:,p], "r.", markersize=4, label="prediction")
  #         plt.legend(loc="best")
  #         plt.xlabel("Period")       
  #         plt.show()
         
         
        
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
        
    
    print("Single Series Predicting",predict_ahead_steps,"steps ahead.")
 
    print("Loading mat_sales_x")
    mat_sales_x=np.load("mat_sales_x.npy")

    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
   
  #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
  #  sales=np.swapaxes(series2,0,2)
  #  print("sales shape=",sales.shape)
  #  n_steps=sales.shape[1]
 
  #  mat_sales=np.swapaxes(mat_sales,0,2)
  #  mat_sales_90=np.swapaxes(mat_sales_90,0,2)

    print("mat_sales_x shape",mat_sales_x.shape,"n_steps=",n_steps)
   
    #print("mat sales shape",mat_sales.shape)

    original_steps=mat_sales_x.shape[1]
    ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
    print("ys shape",ys.shape)

   # print("y value mean shape=",y_value_mean.shape)
   # y_pred_mean= model.predict(y_value_mean[:,start_point:,:])    #[:, np.newaxis,:]
  #  print("y pred mean",y_pred_mean.shape)

    #       y_values=np.stack([model(X_test,training=True) for sample in range(20)])
 #       y_value_mean=y_values.mean(axis=0)
 #       y_value_std=y_values.std(axis=0)

    
    #ys= model.predict(mat_sales_90[:,start_point:,:])    #[:, np.newaxis,:]
    pas=predict_ahead_steps   #,dtype=tf.none)
    
 #   step_ahead=tf.constant(1,dtype=tf.int32)

    for step_ahead in range(1,pas):
          print("\rstep:",step_ahead,"/",pas,"ys shape",ys.shape,"n_steps",n_steps,end='\r',flush=True)
          y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
          ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
     #     print("ys",ys,"ys shape",ys.shape,step_ahead)
         

    print("\rstep:",step_ahead+1,"/",pas,end='\n\n',flush=True)

 #   new1_ys=tf.make_ndarray(tf_ys)
  #  new1_ys=np.array(tf_ys)
#################################################################    
    
 #   mat_swap_sales=np.swapaxes(mat_sales,0,1)
 #   print("mat_swap_sales shape",mat_swap_sales.shape)
 #   print("mat sales shape",mat_sales.shape)
    #      y_values=np.stack([model(X_test,training=True) for sample in range(20)])
 
    # y_value_mean=y_values.mean(axis=0)
    # y_value_std=y_values.std(axis=0)
    
    # print("first pred=",np.round(model.predict(X_test[:1]),2))
    # print("y values=",np.round(y_values[:,:1],2))    
    
    # print("y_value_mean, std=\n",y_value_mean,"\n\n",y_value_std) 
    # print("y_values.shape=",y_values.shape)
    # print("y value mean shape=",y_value_mean.shape)
    # print("y value stddev shape=",y_value_std.shape)
        
    # print('\n# Evaluate y_pred on test data')
    # results = model.evaluate(X_test, y_value_mean, batch_size=4)
    # print('test loss, test acc:', results)

       
    # history = model.fit(X_test, y_value_mean, epochs=epochs_wavenet)
    # print("loss history=",history.history["loss"])
    #    #, callbacks=callbacks,
    #     #                    validation_data=(X_valid, y_valid))

    # print("plot y values learning curve")
    # plot_learning_curves("y values learning curve",epochs_wavenet,history.history["loss"],[])   #, history.history["val_loss"])
    # plt.show()

    # #yerr = np.linspace(2000, 5000, predict_ahead_steps-1)
    # yerr = np.linspace(100, 5000, predict_ahead_steps-1)
    # print("yerr shape=",yerr.shape)
    
    
   # x_axis_numbers=[*range(-mat_sales_x.shape[1], 0), *range(0,predict_ahead_steps)]
   # print("x axis numbers",x_axis_numbers,len(x_axis_numbers))
   
  #   print("\nloading scaler")
  #   # load the scaler
  #   scaler = load(open('scaler.pkl', 'rb'))
    
  #   new_ys=scaler.inverse_transform(ys[0]).reshape(-1,1)
  #   mat_sales_x=scaler.inverse_transform(mat_sales_x[0]).reshape(-1,1)

  #   print("mat sales x=\n",mat_sales_x,mat_sales_x.shape)
  #   print("ys=\n",new_ys,new_ys.shape)
  # #  inverse_scaled=scaler.inverse_transform(scaled_series)
  #   print("\nunscaled ys=\n",ys,ys.shape)

    for p in range(0,n_query_rows):

        plt.title("Series Pred: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
        
      #  plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "b-", markersize=5, label="actual wk")
      #  plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "m-", markersize=5, label="actual mat")
  #      plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "b-", markersize=5, label="actual")
        plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "b-", markersize=5, label="actual")


   #     print("yerr=",yerr)
  #      plt.errorbar(range(sales.shape[1],sales.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps):,p], "r-", markersize=5, yerr=yerr, label="prediction")
        plt.plot(range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")


        plt.plot(range(mat_sales_x.shape[1]-1,mat_sales_x.shape[1]+predict_ahead_steps-1), ys[0,-(predict_ahead_steps):,p], "r-", markersize=5, label="prediction")


     #   plt.plot(range(start_point,ys.shape[1]+start_point), ys[0,:,p], "g.", markersize=5, label="validation")

   #     plt.plot(range(start_point,y_pred_mean.shape[1]+start_point), y_pred_mean[0,:,p], "y-", markersize=5, label="validation")

   #     plt.plot(range(start_point,ys.shape[1]), ys[0,start_point:,p], "b.", markersize=5, label="validation")

 #       plt.plot(range(start_point,sales.shape[1]), ys[0,start_point:sales.shape[1],p], "g.", markersize=5, label="validation")


  #      plt.errorbar(range(sales.shape[1],sales.shape[1]+predict_ahead_steps-1), ys[0,-(predict_ahead_steps-1):,p], yerr=yerr, linestyle="-",label='Prediction + error')      
        plt.legend(loc="best")
        plt.xlabel("Period")
        
        plt.show()
 


#############################################################################33

    print("graphing ys=\n",ys,ys.shape)
    print("extended series table=\n",extended_series_table,extended_series_table.shape)
    predict_series=np.swapaxes(ys[0],0,1)
    print("predict series=\n",predict_series,predict_series.shape)
    extra_series=extended_series_table.iloc[:,:(extended_series_table.shape[1]-predict_series.shape[1])].to_numpy()
    print("extra series=\n",extra_series,extra_series.shape)
    new_ys=np.hstack((extra_series,predict_series))
    print("new ys series=\n",new_ys,new_ys.shape)
    tes=extended_series_table.T
  #  tes.set_index('period',inplace=True)
  
    pred_product_names=[s + "_pred" for s in product_names]   #pred_product_names=product_names+"_pred"
    print("pred prod names",pred_product_names)
    tnys=new_ys.T
    i=0
    for col in pred_product_names:
        print("col=",col,i)
        tes[col]=tnys[:,i]
        i+=1
  #  tnys=pd.DataFrame(new_ys.T,columns=pred_product_names)
  #  print("1tnys series=\n",tnys,tnys.shape)
  #  tnys['period']=extended_dates
  #  tnys.set_index('period',inplace=True)
   # print("2tnys series=\n",tnys,tnys.shape)
    #  new_index=['period']
  #  new_index=['period'] #+product_names
    #new_index=new_index.flatten()
#    print("new index=",new_index)
 #   tnys['period']=tnys.reindex(new_index)
 #   tnys['period'] = tnys['period'].astype('category')
#    print("2tnys series=\n",tnys,tnys.shape,tnys.shape)
  #  print("tes extended series table=\n",tes,tes.shape,tes.index)
  #  print("prod names=",product_names)
  #  print("prod names+str=",product_names+list(np.str("-pred")))
   
 #   new_tes=pd.concat([tes,tnys],axis=1)   #.reindex(tes.index)   #,ignore_index=True)
    #new_tes=tes.merge(tnys,how='left',on=['period','period'])   #,left_on='period',right_on='period',left_index=True,right_index=False)
  
    #new_tes.set_index('period',inplace=True)  
  
 #   print("new predict series=\n",new_tes,new_tes.shape)
 #   tes.set_index('period',inplace=True)
   # print("inddexed new predict series=\n",tes,tes.shape,tes.index)
    tes=tes.reindex(natsorted(tes.columns),axis=1)   # sort by column name
    extended_product_names=tes.columns
    extended_series_table=tes.T
    print("end extended series table=\n",extended_series_table,extended_series_table.shape)
  #  print("end extended series table[0:2]=\n",extended_series_table[2:3])

   # for p in range(0,extended_series_table.shape[0])
   # tes[]

    
#############################################################################3

    for p in range(0,extended_series_table.shape[0],2):
        plt.figure(figsize=(11,4))
        plt.subplot(121)
        plt.title("A unit sales prediction: "+str(extended_product_names[p]),fontsize=14)
      
        plt.ylabel("Units")
        plt.xlabel("Period") 
        graph_a_prediction(extended_series_table[p:p+2],extended_dates,p)
       # graph_a_series(extended_series_table[p+1:p+2],extended_dates,p+1,True)
       
        plt.legend(loc="best")
        plt.show()
         
  
    
    
    
    
 ############################################################################3
     #  MCDropout predict
        # create a new testing X batch
    if False:        
        print("mat sales x shape=",mat_sales_x.shape)
        n_steps=mat_sales_x.shape[1]
            
        
        print("\nMCDropout improved Series Predicting",predict_ahead_steps,"steps ahead.")
        
        model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})
        n_steps=mat_sales_x.shape[1]
     
        print("mat_sales_x shape",mat_sales_x.shape)
       
        print("mat sales x",mat_sales_x)
    
        for step_ahead in range(7):
    #          y_probas=np.stack([model(X_test,training=True) for sample in range(3)])
              y_probas=np.stack([model(X_test,training=True) for sample in range(3)])
    
              y_mean=np.round(y_probas.mean(axis=0),2)
              y_stddev=np.round(y_probas.std(axis=0),2)
        #      print("sahead loop mat sales 90[0,:]=",mat_sales_90[0,:],mat_sales_90.shape)
              print("sahead loop y mean[:,-1]=",y_mean[:,-1],y_mean.shape)
              mat_sales_x=np.append(mat_sales_x[0,:],y_mean[:,-1]).reshape(1,-1,n_query_rows)
              print("saehad loop mat sales 90=",mat_sales_x.shape,"n steps=",n_steps)
              X_test,y=build_mini_batch_input(mat_sales_x[-n_steps:],10000,batch_length)
              #X_test=np.stack(X_test,new_X_test)
          #    print("step ahead=",step_ahead,"y_mean=\n",y_mean[:,-1,:],"y_mean shape=",y_mean.shape)  #[:, np.newaxis,:]
              print("sahead",step_ahead,"sahead loop appended X test shape",X_test.shape)
     
        yerr = np.linspace(0, 0, predict_ahead_steps)
        print("yerr shape=",yerr.shape)
        
        
        
        for p in range(0,n_query_rows):
    
            plt.title("MCDropout Series Prediction: Actual vs Prediction: "+str(product_names[p]),fontsize=14)
            
            plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "b-", markersize=5, label="actual wk")
            plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "m-", markersize=5, label="actual mat")
            plt.plot(range(0,mat_sales_x.shape[1]), mat_sales_x[0,:,p], "r-", markersize=5, label="actual 28 day MT")
    
            plt.plot(range(mat_sales_x.shape[1],mat_sales_x.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps):,p], "r.", markersize=5, label="prediction")
    
       #     print("yerr=",yerr)
      #      plt.errorbar(range(sales.shape[1],sales.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps):,p], "r-", markersize=5, yerr=yerr, label="prediction")
    
            plt.plot(range(start_point+1,ys.shape[1]+start_point+1), ys[0,:,p], "g.", markersize=5, label="validation")
    
       #     plt.plot(range(start_point,y_pred_mean.shape[1]+start_point), y_pred_mean[0,:,p], "y-", markersize=5, label="validation")
    
       #     plt.plot(range(start_point,ys.shape[1]), ys[0,start_point:,p], "b.", markersize=5, label="validation")
    
     #       plt.plot(range(start_point,sales.shape[1]), ys[0,start_point:sales.shape[1],p], "g.", markersize=5, label="validation")
    
    
            plt.errorbar(range(mat_sales_x.shape[1],mat_sales_x.shape[1]+predict_ahead_steps), ys[0,-(predict_ahead_steps):,p], yerr=yerr, linestyle="-",label='Prediction + error')      
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
    print("series table shape=",series_table.shape)
    print(series_table)
    print("ys",ys.shape)    
    
    d=len(dates)
    
    swap_ys=np.swapaxes(ys,0,2)[:,-predict_ahead_steps:,0]
    
    print("swap_ys 1",swap_ys.shape)    
   # swap_ys=swap_ys[:,:,0]
   # print("swap_ys 2",swap_ys.shape)    
   
    pred_df1=pd.DataFrame(data=swap_ys,   #,    # values
             index=np.array(product_names))  #,   # 1st column as index
            # columns=ddates)  # 1st row as the column names
    
    print("pd=",pred_df1)
  #  print("y=",y)
    print("First date=",dates[0])
    print("last date=",dates[-1])
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
        series_table.T.to_csv(f)  #,line_terminator='rn')
        print("table.T.shape=",series_table.T.shape)
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


    
          
          
          
