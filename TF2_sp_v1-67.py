# -*- coding: utf-8 -*-
"""

@author: Anthony Paech 2016
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:29:55 2020
"""



# import os


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
import joblib
import pickle
from natsort import natsorted
from pickle import dump,load
import datetime as dt
from datetime import date

from sklearn.preprocessing import StandardScaler,MinMaxScaler

#import itertools
from natsort import natsorted
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

    
def load_data(mats,filename,index_code):    #,col_name_list,window_size):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows
   
    df.fillna(0,inplace=True)
    
  #  print(df['productgroup'])
#    df['productgroup']=(df['productgroup']).astype(int)
 #   df['productgroup']=df['productgroup'].astype(str)

    #  print(df)

 #   print("mask passer=",mask_passer)
  #  print("df.mask",df[mask_passer])
  #  print("df.mask(bool)",df[bool(mask_passer)])

#    print("df.mask(bool",df.mask(bool(mask_passer)))
    
 #   df["period"]=df.date.dt.to_period('W')
   
    df["period"]=df.date.dt.to_period('D')
   # df["period"]=df.date.dt.to_period('B')


    df=create_margins_col(df)

 #   print("df=\n",df,df.shape)
    
       #     dates=list(set(list(series_table.T['period'])))   #.dt.strftime("%Y-%m-%d"))))
  #  dates=list(set(list(df['period'].dt.strftime("%Y-%m-%d"))))   #.dt.strftime("%Y-%m-%d"))))

  #  dates.sort()

  #  df["period"]=df.date.dt.to_period('B')    # business days  'D' is every day
  #  df["period"]=df.date.dt.to_period('D')    # business days  'D' is every day
   #  periods=set(list(df['period']))
  #  dates=list(df['period'].dt.strftime("%Y-%m-%d"))
  #  dates.sort()
 #   mask = mask.replace('"','').strip()    
 #   print("mask=",mask)
       # mask=(df['product']=='SJ300')
   # mask=((df['code']=='FLPAS') & (df['product']=='SJ300'))
    mask=((df['code']=='FLPAS') & (df['productgroup']==10) & (df['product']=='SJ300'))   #mask=bool(mask_str)
  #  mask=((df['productgroup']>=10) & (df['productgroup']<=11))
  #  mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='TS300')))
  #  mask=(df['cat']=='77')

  #  index_code=['product']  
  #  index_code=['code','product']
 #   index_code=['code','productgroup','product']

   # mask=(df['product']=='SJ300')
   # mask=((df['code']=='FLPAS') & (df['product']=='SJ300'))
  #  mask=((df['code']=='FLPAS') & (df['productgroup']==10) & (df['product']=='SJ300'))
  #  mask=((df['productgroup']>=10) & (df['productgroup']<=11))
  #  mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='TS300')))
  #  mask=(df['cat']=='77')
  #  mask=(df['code']=='FLPAS')
  #  mask=((df['code']=='FLPAS') & (df['product']=="SJ300") & (df['glset']=="NAT"))
#    df['productgroup'] = df['productgroup'].astype('category')
 #   mask=((df['productgroup']>=10) & (df['productgroup']<=14))
 #   mask=bool(mask_str)   #((df['productgroup']>=10) & (df['productgroup']<=14))"

 #   mask=(df['productgroup']==13)
 #   mask=((df['code']=='FLPAS') & (df['product']=="SJ300"))
  
 #   mask=((df['code']=='FLPAS') & ((df['product']=="CAR280") | (df['product']=="SJ300")))
  #  mask=((df['code']=='FLPAS') & (df['productgroup']==10))  # & ((df['product']=='SJ300') | (df['product']=='AJ300')))
 #   mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='AJ300') | (df['product']=='TS300')))

   # print("mask=",str(mask))
    print("pivot table being created.")
  #  table = pd.pivot_table(df[mask], values='qty', index=['code','productgroup','product'],columns=['week'], aggfunc=np.sum, margins=False,observed=False, fill_value=0)   #observed=True
  #  table = pd.pivot_table(df[mask], values='qty', index=['code','product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
  #  table = pd.pivot_table(df[mask], values=['qty'], index=['productgroup'],columns=['period'], aggfunc=[np.sum], margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
  #  table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0)   #observed=True
    
    dates=list(set(list(df['date'].dt.strftime("%Y-%m-%d"))))

    dates.sort()
    
    df['period'] = df['period'].astype('category')
 
  #  mask=df.mask(bool(mask_passer))
 
    
  #  index_code=['code','productgroup','product']
  #  index_code=['code','product']

 
#################################################################

    print("Making pivot table on unit sales...")
    mat_type="u"   #  "u" "d","m"   # unit, dollars, margin

    table_u = pd.pivot_table(df[mask], values='qty', index=index_code,columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True

    colnames=list(table_u.columns)
    print("colnames_u=\n",colnames)
      
    for window_length in range(0,len(mats)):
        col_no=0
        for col in colnames:
            table_u.rename(columns={col: str(col)+"@1#"+str(mat_type)},inplace=True)
            table_u=add_mat(table_u,col_no,col,mats[window_length],mat_type)
            col_no+=1
  
################################################33
 

    if False:    #  extra series
        print("Making pivot table on dollar sales...")
    
        mat_type="d"   #  "u" "d","m"   # unit, dollars, margin
    
        table_d = pd.pivot_table(df[mask], values='salesval', index=index_code,columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True
     
        colnames=list(table_d.columns)
      #   print("colnames_d=\n",colnames)
        for window_length in range(0,len(mats)):
            col_no=0
            for col in colnames:
                table_d.rename(columns={col: str(col)+"@1#"+str(mat_type)},inplace=True)
                table_d=add_mat(table_d,col_no,col,mats[window_length],mat_type)
                col_no+=1
     
    ####################################################
        print("Making pivot table on margins...")
    
        table_1=table_u.merge(table_d,on='period',how='left')    
       
        mat_type="m"   # "u" "d","m"   # unit, dollars, margin
    
        table_m = pd.pivot_table(df[mask], values='margin', index=index_code,columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True
    
        colnames=list(table_m.columns)
      #  print("colnames_d=\n",colnames)
        for window_length in range(0,len(mats)):
            col_no=0
            for col in colnames:
                table_m.rename(columns={col: str(col)+"@1#"+str(mat_type)},inplace=True)
                table_m=add_mat(table_m,col_no,col,mats[window_length],mat_type)
                col_no+=1
     
    
    
        table=table_1.merge(table_m,on='period',how='left')   
        table = table.reindex(natsorted(table.columns), axis=1) 

    else:
        table = table_u.reindex(natsorted(table_u.columns), axis=1) 

    
    if len(mats)>0:
     #   table=remove_series_without_str(table,"@1#")
        print("remove @1#, mats=",mats)     
        table=remove_a_series_subset(table,"@1#")   #if other mats, delete original series
    
    table=filter_series_on_str(table,"@")   #if other mats, delete original series

 #   print("final concat table=\n",table,table.shape) 
 #   print("final concat table.T=\n",table.T,table.T.shape) 

    return table,dates


def create_margins_col(df):
    # add a column of salesval-costval for every transaction
    df['margin']=df['salesval']-df['costval']
#    df['gross_margin']=round(df['margin']/df['salesval'],3)
    return df



def add_mat(table,col_no,col_name,window_period,mat_type):
    start_mean=table.iloc[:window_period,col_no].mean(axis=0) 
    
#  print("start mean=",window_period,start_mean)   # axis =1
    mat_label=str(col_name)+"@"+str(window_period)+"#"+str(mat_type)+":mt_" 
    table[mat_label]= table.iloc[:,col_no].rolling(window=window_period,axis=0).mean()
    return table.fillna(start_mean)


def filter_series_on_str(table,mask_str):
    return table.filter(like=mask_str,axis=0)


def remove_a_series_subset(table,mask_str):   #,col_name_list,window_size):
    if table.index.nlevels>1: 
        if len(table.columns)>1:
            tc = [''.join(col).strip() for col in table.columns.values]
            print("len>1 multi index tc=",tc)
            mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]
      #      table=table.T
     #       table=table[mask]

        else:
            tc=list(table.columns.values)
            print("len <=1 multi index tc=",tc)
           # mask=(mask_str not in tc)
            mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]

    #        table=table.T
    else:    
        tc=list(table.columns.values) 
        print("single index tc=",tc)
        mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]
      #  mask=(mask_str not in tc)
   #     table=table.T

    table=table.T
    table=table[mask]

    #mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]
    #table=table.T
    #table=table[mask]
    return table

  

def convert_pivot_table_to_numpy_series(series_table):

#     mat_sales=series_table.iloc[row_no_list,:].to_numpy()
     mat_sales=series_table.to_numpy()

     mat_sales=np.swapaxes(mat_sales,0,1)
     mat_sales=mat_sales[np.newaxis] 
     return mat_sales



def build_mini_batch_input(series,no_of_batches,no_of_steps):
    print("build mini batch series shape",series.shape)
    np.random.seed(42) 
    series_steps_size=series.shape[1]

    random_offsets=np.random.randint(0,series_steps_size-no_of_steps,size=(no_of_batches)).tolist()
    new_mini_batch=series[:,random_offsets[0]:random_offsets[0]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)

   # prediction=new_mini_batch[:,-1]
    for i in range(1,no_of_batches):
        temp=new_mini_batch[:,:no_of_steps+1]
        new_mini_batch=series[:,random_offsets[i]:random_offsets[i]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)
    #    prediction=np.concatenate((prediction,new_mini_batch[:,-1]))
        new_mini_batch=np.vstack((temp,new_mini_batch[:,:no_of_steps+1]))
        
        if i%100==0:
            print("\rBatch:",i,"new_mini_batch.shape:",new_mini_batch.shape,flush=True,end="\r")
    return new_mini_batch[:,:no_of_steps,:],new_mini_batch[:,1:no_of_steps+1,:]




def graph_whole_pivot_table(series_table,dates): 
    np.random.seed(43) 
    series_table=series_table.T  
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
    ax = plt.gca()
    ax.tick_params(axis = 'x', which = 'major',rotation=45, labelsize = 8)

#    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
#    ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
    cols=list(series_table.columns)
 #   print("1cols=",cols)
    del cols[-1]  # delete reference to period column
  #  print("2cols=",cols)
   
  #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
    #colors = cm.rainbow(np.linspace(0, 2, len(cols)))
    plt.legend(loc="best",fontsize=8)
    plt.title("Sales per day",fontsize=9)
    plt.ylabel("Units or $",fontsize=9)
    plt.xlabel("Date",fontsize=8) 

    for col in cols:
        color=np.random.rand(len(cols),3)
        series_table.plot(kind='line',x='period',y=col,color=color,ax=ax,fontsize=8)

    plt.show()
  #  print("graph finished")
    return 
 


def extend_pivot_table(series_table,dates,product_names,periods_len): 
    first_date=series_table.T.index[0].strftime('%Y-%m-%d')
    last_date=series_table.T.index[-1].strftime('%Y-%m-%d')
    
    len_product_names=len(product_names)
    pidx = pd.period_range(first_date, periods=periods_len)   # 2000 days
    #print("series_table=\n",series_table,series_table.index)
    new_table = pd.DataFrame(np.nan, index=product_names,columns=pidx)   #,dtype='category')  #series_table.columns)
    new_table=new_table.T
 #   print("new table=\n",new_table,new_table.columns)
   
    series_table=series_table.T
    extended_table=series_table.join(new_table,how='outer',rsuffix="_r?",sort=True) 
    extended_table.drop(extended_table.columns[range(len_product_names,len_product_names*2)],axis=1,inplace=True)
    exdates=extended_table.index.astype(str).tolist()  #.astype(str)) #strftime("%Y-%m-%d"))
 #   print(" extend table ->extended_table.T=\n",extended_table,extended_table.shape)  #[:4,:4].to_string())

    return extended_table.T,exdates
 
    
    
def actual_days(series_table):
    first_date=series_table.T.index[0].to_timestamp(freq="D",how="S")
    last_date=series_table.T.index[-1].to_timestamp(freq="D",how="S")
    return (last_date - first_date).days    #.timedelta_series.dt.days    

    
  
def add_a_new_series(table,arr_names,arr,start_series_len,predict_ahead_steps,periods_len):
    first_date=table.T.index[0].strftime('%Y-%m-%d')
    pidx = pd.period_range(first_date, periods=periods_len)   # 2000 days  
    new_cols=pd.DataFrame(arr[0],columns=arr_names,index=pidx)
    table=table.T 
    table2=pd.concat((table,new_cols),join='outer',axis=1)   
    new_product_names=table2.columns
    table2=table2.T
    return table2,new_product_names
    
     
        

def find_series_type(series_name):
    return series_name[series_name.find(':')+1:]

   

def graph_a_series(series_table,dates,column_names): 

 #   series_dict_elem=series_dict_elem.astype(str)  
    series_table = series_table.reindex(natsorted(series_table.columns), axis=1)
    series_table=series_table.T  
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
 #   print("\ngraph a series, table.T=\n",series_table,series_table.shape)
 
#    series_table=np.num_to_nan(series_table,0)
 #   series_table[column_names] = series_table[column_names].replace({0:np.nan})
 #   print("graph a series - series_table",series_table)
    ax = plt.gca()
    cols=list(series_table.columns)
    del cols[-1]  # delete reference to period column
    col_count=0
    for col in cols:
        
        series_table[col] = series_table[col].replace({0:np.nan})
  #      print("\ngraph a series - series_table",col,"\n",series_table[col])
  
      #  print("find series type",col,"=",find_series_type(col))  
        series_suffix= str(find_series_type(col)) 
     #   print("series suffix=",series_suffix)
      #  series_type=str(series_dict[series_suffix])   # name, type of plot, colour
   #     print("series type=\n",series_type,">",series_type)   # name, type of plot, colour
        if (series_suffix=="mt_pred_mc"): # | (series_suffix=="mt_yerr_mc")):
            pred_plot=col
      #      print("pred_polt=",pred_plot)

        if col in column_names:
#            series_table.plot(kind=series_dict_elem[1],x='period',y=col,color=series_dict_elem[2],ax=ax,fontsize=8)
             #    plt.errorbar('period', series_table[col], yerr=series_table.iloc[col_count+1], data=series_table)
            if series_suffix=="mt_yerr_mc":
       #         print("\nplotting error bar\n")
                plt.errorbar('period', pred_plot, yerr=col, fmt="r.",data=series_table,ecolor="magenta",errorevery=2)
 
            else:   
                if series_suffix=="mt_":
                     plt.plot(series_table['period'],series_table[col],"b.",markersize=3,label=col)    #,range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
                elif series_suffix=="mt_pred_mc":        
                    plt.plot(series_table['period'],series_table[col],"g.",markersize=3,label=col) 
                else: 
                    pass
                 #   plt.plot(series_table['period'],series_table[col],"r-",markersize=3,label=col) 
             #   series_table.plot(kind='scatter',x='period',y=col,color=series_type,ax=ax,fontsize=8,s=2,legend=False)
              #      series_table.plot(kind='line',x='period',y=col,color=series_type,ax=ax,fontsize=8)

        col_count+=1    

    return 
    
  




# def plot_learning_curves(title,epochs,loss, val_loss):
#     plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
#     plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
#     plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
#     plt.axis([1, epochs, 0, np.amax(loss)])
#     plt.legend(fontsize=11)
#     plt.title(title,fontsize=11)
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.grid(True)
#     return


def plot_log_learning_curves(title,epochs,loss, val_loss):
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

    predict_ahead_steps=1000

 #   epochs_cnn=1
    epochs_wavenet=60
    no_of_batches=50000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
#    y_length=1
    neurons=1400
    start_point=150
    pred_error_sample_size=20
    
    # series_dict=dict({"mt":"blue",
    #                   "mt_":"blue",
    #                   "pred":"red",
    #                   "mt_pred":"red",
    #                   "pred_mc":"green",
    #                   "mt_pred_mc":"green",
    #                   "yerr_mc":"green",
    #                   "mt_yerr_mc":"green"
    #                   })
    
   
    # train validate test split 
    train_percent=0.7
    validate_percent=0.2
    test_percent=0.1
    
    
    filename="NAT-raw310120_no_shop_WW_Coles.xlsx"
       #     filename="allsalestrans020218-190320.xlsx"   
    
     #  index_code=['product']  
  #  index_code=['code','product']
    index_code=['code','productgroup','product']

  
    
    
    
    mats=[7,30,90,365]   #omving average window periods for each data column to add to series table
    


########################################################################    

#    load the sales_trans.xls file
# load the query.xls file
#    this contains all the products, product groups, customers, customer groups, glsets and special price categories you want to     

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors


    answer=input("Rebuild batches?")
    if answer=="y":
 
#    mask="(df['product']=='SJ300')"
     #   filename=ic.filename    #"NAT-raw310120all.xlsx"
   # #     filename="allsalestrans020218-190320.xlsx"
    
      #  mats=ic.mats   #[5,65]    # series moving average window periods for each data column to add to series table
      #  col_name_list=["10","14"]
      #  window_size=90
        
     
    #    filename=ic.filename   #"shopsales020218to070320.xlsx"
    
        print("loading data....",filename) 
    #   series2,product_names,periods=load_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
     #   series2,product_names,ddates,table,mat_table,mat_sales,mat_sales_90=load_shop_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
        series_table,dates=load_data(mats,filename,index_code)    #,col_name_list,window_size)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)

     #   series_table=series_table.T
     
        if series_table.index.nlevels>1: 
            series_table.index = series_table.index.map(''.join).astype(str) 
     #   series_table=series_table.T
        
        actual_days_in_series_table=actual_days(series_table)
        
        
        product_names=list(series_table.index) 
        print("\nProduct names, length=",product_names,len(product_names))

        periods_len=actual_days_in_series_table+predict_ahead_steps+1
        print("PERIODS=",periods_len)

   
        
        #  this adds 
        graph_whole_pivot_table(series_table,dates)
        
        extended_series_table,extended_dates=extend_pivot_table(series_table,dates,product_names,periods_len)
        
      #  graph_whole_pivot_table(extended_series_table,extended_dates)
        
     
        print("Saving pickled table - series_table.pkl")
        pd.to_pickle(series_table,"series_table.pkl")
        print("Saving pickled extended_series_table - extended_series_table.pkl")
        pd.to_pickle(extended_series_table,"extended_series_table.pkl")

        extended_series_table.to_csv("extended_series_table.csv")     

        print("Saving product names")
        np.save("product_names.npy",np.asarray(product_names))
        
        print("Saving dates",len(extended_dates))
        with open('dates.pkl', 'wb') as f:
            pickle.dump(dates,f)   
        with open('extended_dates.pkl', 'wb') as f:
            pickle.dump(extended_dates,f)   
 
     #   scaled_series=series_table.to_numpy()
        scaled_series=extended_series_table.to_numpy()
        scaled_series=scaled_series[:,:actual_days_in_series_table]
        scaled_series=np.nan_to_num(scaled_series,0)
        
      
     #   print("scaled series=",scaled_series,scaled_series.shape)
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
        
        actual_days_in_series_table=actual_days(series_table)
               
#        product_names=list(series_table.index) 
#        print("\nProduct names, length=",product_names,len(product_names))

        periods_len=actual_days_in_series_table+predict_ahead_steps+1
        print("PERIODS=",periods_len)

   
    
           
    
    print("\n\nseries table loaded shape=",series_table.shape,"\n")  
    print("product names=\n",product_names)
  #  print("series table with date=\n",series_table_with_date)
  #  print("dates array=\n",len(dates))
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
        

   # print("epochs_cnn=",epochs_cnn)
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
   
    X_train, y_train = X[:train_size, :,:], y[:train_size,-1:,:]
    X_valid, y_valid = X[train_size:train_size+validate_size, :,:], y[train_size:train_size+validate_size,-1:,:]
    X_test, y_test = X[train_size+validate_size:, :,:], y[train_size+validate_size:,-1:,:]
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
          print("\rstep:",step_ahead,"/",pas,"ys shape",ys.shape,end='\r',flush=True)
          y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
          ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
     #     print("ys",ys,"ys shape",ys.shape,step_ahead)
         

    print("\rstep:",step_ahead+1,"/",pas,end='\n\n',flush=True)
    pred_product_names=[s + "pred" for s in original_product_names]
  #  print("pred product names=\n",pred_product_names)
#    print("est before=",extended_series_table,extended_series_table.shape)  
#
#   Pad ys array with nans to fit it easily into the extended table.  Add startpoint
  #  print("ys before start padding=\n",ys,ys.shape)
    pad_before_arr = np.empty((1,start_point,ys.shape[2]))
    pad_before_arr[:] = np.NaN
#    print("pad before arr=\n",pad_before_arr,pad_before_arr.shape)

#    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
    ys = np.concatenate((pad_before_arr,ys),axis=1) 
 #   print("ys after the before padding=\n",ys,ys.shape)

 #   print("periods_len=",periods_len,"start point",start_point)
 #   print("ys before end padding=\n",ys,ys.shape)
    pad_after_arr = np.empty((1,periods_len-ys.shape[1],ys.shape[2]))
    pad_after_arr[:] = np.NaN
  #  print("pad after arr=\n",pad_after_arr,pad_after_arr.shape)

#    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
    ys = np.concatenate((ys,pad_after_arr),axis=1) 
  #  print("ys after end  padding=\n",ys,ys.shape)



    extended_series_table,product_names=add_a_new_series(extended_series_table,pred_product_names,ys,start_point,predict_ahead_steps,periods_len)
 #   print("est after=",extended_series_table,extended_series_table.shape)  

   # print("est=",extended_series_table.shape)

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

 #   print("mat_sales_x shape",mat_sales_x.shape,"n_steps=",n_steps)
 


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
          print("\rstep:",step_ahead,"/",pas,"mc_ys shape",mc_ys.shape,end='\r',flush=True)
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
   

   #########################################33 
   
    pred_product_names=[s + "pred_mc" for s in original_product_names]
    
    #   Pad mc_ys array with nans to fit it easily into the extended table.  Add startpoint
  #  print("mc_ys before padding=\n",mc_ys,mc_ys.shape)
    pad_before_arr = np.empty((1,start_point,mc_ys.shape[2]))
    pad_before_arr[:] = np.NaN
  #  print("pad before arr=\n",pad_before_arr,pad_before_arr.shape)

#    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
    mc_ys = np.concatenate((pad_before_arr,mc_ys),axis=1) 
  #  print("mc_ys after the before padding=\n",mc_ys,mc_ys.shape)

    pad_after_arr = np.empty((1,periods_len-mc_ys.shape[1],mc_ys.shape[2]))
    pad_after_arr[:] = np.NaN
  #  print("pad after arr=\n",pad_after_arr,pad_after_arr.shape)

#    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
    mc_ys = np.concatenate((mc_ys,pad_after_arr),axis=1) 
  #  print("mc_ys after end  padding=\n",mc_ys,mc_ys.shape)

    extended_series_table,product_names=add_a_new_series(extended_series_table,pred_product_names,mc_ys,start_point,predict_ahead_steps,periods_len)

#############################################################

    pred_product_names=[s + "yerr_mc" for s in original_product_names]  
    
    
    #   Pad mc_yerr array with nans to fit it easily into the extended table.  Add startpoint
  #  print("mc_yerr before padding=\n",mc_yerr,mc_yerr.shape)
    pad_before_arr = np.empty((1,start_point,mc_yerr.shape[2]))
    pad_before_arr[:] = np.NaN
  #  print("pad before arr=\n",pad_before_arr,pad_before_arr.shape)

#    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
    mc_yerr = np.concatenate((pad_before_arr,mc_yerr),axis=1) 
  #  print("mc_yerr after the before padding=\n",mc_yerr,mc_yerr.shape)

    pad_after_arr = np.empty((1,periods_len-mc_yerr.shape[1],mc_yerr.shape[2]))
    pad_after_arr[:] = np.NaN
  #  print("pad after arr=\n",pad_after_arr,pad_after_arr.shape)

#    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
    mc_yerr = np.concatenate((mc_yerr,pad_after_arr),axis=1) 
  #  print("mc_yerr after end  padding=\n",mc_yerr,mc_yerr.shape)
    
    
    extended_series_table,product_names=add_a_new_series(extended_series_table,pred_product_names,mc_yerr,start_point,predict_ahead_steps,periods_len)


##############################################################


    for p in range(0,extended_series_table.shape[0]):
        plt.figure(figsize=(11,4))
        plt.subplot(121)
        
        ax = plt.gca()
    #    ax.tick_params(axis = 'both', which = 'major',rotation=90, labelsize = 6)
        ax.tick_params(axis = 'x', which = 'major',rotation=45, labelsize = 8)

    #       ax.tick_params(axis = 'both', which = 'minor', rotation=90, labelsize = 6)

        plt.title("Sales/day:Actual+Pred: "+str(product_names[p]),fontsize=10)
      
        plt.ylabel("Units or $",fontsize=9)
        plt.xlabel("Date",fontsize=9) 
        graph_a_series(extended_series_table,extended_dates,product_names[p])
        
        plt.legend(loc="best",fontsize=8)
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



       
#######################################################################################    
    
    print("\nwrite predictions to sales_prediction.CSV file....")
#    dates.sort()
    extended_series_table=extended_series_table.T
   # print("extended series table=\n",extended_series_table)
    
    print("\nextended series table shape=",extended_series_table.shape)

    print("First date=",extended_dates[0])
    print("last date=",extended_dates[-1])
    with open("sales_prediction_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
        extended_series_table.to_csv(f)  #,line_terminator='rn')
    
    print("\n\nFinished.")
     
    return


if __name__ == '__main__':
    main()


    
          
          
          
