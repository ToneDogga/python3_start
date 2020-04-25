# -*- coding: utf-8 -*-
"""

@author: Anthony Paech 2016
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:29:55 2020
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
print("keras:",keras.__version__)
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
from datetime import timedelta

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




###########################################################

# print("\ntensorboard log start")

# root_logdir = os.path.join(os.curdir, "my_logs")

# def get_run_logdir():
#     import time
#     run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
#     return os.path.join(root_logdir, run_id)

# run_logdir = get_run_logdir()
# print("run log dir",run_logdir)




# help(keras.callbacks.TensorBoard.__init__)





##############################################################3

    
def load_data(filename,index_code,mat_type_dict):    #,col_name_list,window_size):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows
   
    df.fillna(0,inplace=True)
        
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
  #  mask=((df['code']=='FLNOR') & (df['productgroup']==10))   #& (df['product']=='SJ300'))   #mask=bool(mask_str)
  #  mask=((df['productgroup']>=10) & (df['productgroup']<=11))
 #   mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='TS300')))
  #  mask=(df['cat']=='77')

  #  mask=(df['productgroup']==13) 
  #  index_code=['product']  
  #  index_code=['code','product']
 #   index_code=['code','productgroup','product']
#    mask=((df['product']=="CCP240") & (df['cat']==77))
    mask=(df['product']=="TS300") 


  #  mask=((df['productgroup']==11) & (df['code']=="FONTANA"))
 #   mask=((df['code']=='FLPAS') & ((df['productgroup']==10) | (df['productgroup']==11)))
  #  mask=((df['code']=='FLPAS') & (df['productgroup']==10) & (df['product']=='SJ300'))
    
  #  mask=((df['code']=='FLPAS') & (df['productgroup']==10) & ((df['product']=='SJ300') | (df['product']=='OM300')))
  #  mask=(df['productgroup']==13) 

  #  mask=((df['productgroup']>=10) & (df['productgroup']<=11))
  #  mask=((df['code']=='FLPAS') & ((df['product']=='SJ300') | (df['product']=='TS300')))
  #  mask=(df['cat']=='77')
  #  mask=((df['code']=='FLPAS'))   # | (df['code']=='FLFUL') |(df['code']=='FONTANA'))
  #  mask=((df['code']=='FLPAS') & (df['product']=="SJ300") & (df['glset']=="NAT"))
#    df['productgroup'] = df['productgroup'].astype('category')
 #   mask=((df['productgroup']>=10) & (df['productgroup']<=14))
 #   mask=bool(mask_str)   #((df['productgroup']>=10) & (df['productgroup']<=14))"

 #   mask=(df['productgroup']==13)
 #   mask=(df['code']=='FONTANA')
  
 #   mask=((df['code']=='FLPAS') & ((df['product']=="CAR280") | (df['product']=="SJ300")))
  #  mask=((df['code']=='FLPAS') & ((df['productgroup']>=10) & (df['productgroup']<=15))) 
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
 
    
 ##################################################
    # print("mat type dict",mat_type_dict)
    # print("keys",mat_type_dict.keys)
    # print("values",mat_type_dict.values)
    # for m in list(mat_type_dict.keys()):
    #     print("k",m,"v",mat_type_dict.get(m)[1])
    
    
    
    print("Making pivot table on unit sales...")
  #  print("Making pivot table on dollar sales...")

  #  mat_type="u"
 
    table = pd.pivot_table(df[mask], values='qty', index=index_code,columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True

   #     colnames=list(table_m.columns)
    #   #  print("colnames_d=\n",colnames)
    #     for window_length in range(0,len(mats)):
    #         col_no=0
    #         for col in colnames:
    #             table_m.rename(columns={col: str(col)+"@1#"+str(mat_type)},inplace=True)
    #             table_m=add_mat(table_m,col_no,col,mats[window_length],mat_type)
    #             col_no+=1
   


# ################################################33
 

# #     #if False:    #  extra series
#     print("Making pivot table on dollar sales...")

# #     mat_type="d"   #  "u" "d","m"   # unit, dollars, margin

#     table_d = pd.pivot_table(df[mask], values='salesval', index=index_code,columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True
 
 
# # ####################################################
#     print("Making pivot table on margins...")

#     table_1=table_u.merge(table_d,on='period',how='left')    
   
# #     mat_type="m"   # "u" "d","m"   # unit, dollars, margin

#     table_m = pd.pivot_table(df[mask], values='margin', index=index_code,columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True




#     table=table_1.merge(table_m,on='period',how='left')   
#   #  table = table.reindex(natsorted(table.columns), axis=1) 

    # else:
    
    return table,dates




def create_margins_col(df):
    # add a column of salesval-costval for every transaction
    df['margin']=df['salesval']-df['costval']
#    df['gross_margin']=round(df['margin']/df['salesval'],3)
    return df





def add_all_mats(series_table,mats,pivot_len,mat_types):
    series_table=series_table.T

    colnames=list(series_table.columns)
 #   print("colnames_u=\n",colnames)
     
  #  mat_type=["u","d","m"]    #  "u" "d","m"   # unit, dollars, margin

    for mat in mat_types:
        for window_length in range(0,len(mats)):
            col_no=0
            for col in colnames:
                series_table.rename(columns={col: str(col)+"@1#"+str(mat)},inplace=True)
                series_table=add_mat(series_table,col_no,col,mats[window_length],mat,pivot_len)
                col_no+=1
  

    series_table = series_table.reindex(natsorted(series_table.columns), axis=1) 

 #   print("it series_table=\n",series_table,series_table.shape)
    if len(mats)>0:
      #  table=remove_series_without_str(table,"@1#")
    #    print("remove @1#, mats=",mats)     
        series_table=remove_a_series_subset(series_table,"@1#")   #if other mats, delete original series
  #  print("1final all mats series table=\n",series_table,series_table.shape) 
   
    series_table=filter_series_on_str(series_table,"@")   #if other mats, delete original series

  #  print("2final all mats series table=\n",series_table,series_table.shape) 
  #  print("final all mats series table.T=\n",series_table.T,series_table.T.shape) 

    return series_table





def add_mat(table,col_no,col_name,window_period,mat_type,pivot_len):
    start_mean=table.iloc[:window_period,col_no].mean(axis=0) 
    
#  print("start mean=",window_period,start_mean)   # axis =1
    mat_label=str(col_name)+"@"+str(window_period)+"#"+str(mat_type)+":mt_" 
 #   print("table.iloc[:,col_no]",table.iloc[:,col_no])
 #   print("table.iloc[:pivot_len,col_no]",table.iloc[:pivot_len,col_no])
    table.iloc[:pivot_len,col_no].fillna(0,inplace=True) 
#    table[mat_label]= table.iloc[:pivot_len,col_no].rolling(window=window_period,axis=0).mean()
    table[mat_label]= table.iloc[:pivot_len,col_no].rolling(window=window_period,axis=0).mean()
 #   table.iloc[:,pivot_len] = table.iloc[:,pivot_len].ffill()
#    table.iloc[:,:pivot_len].ffill(start_mean)
#    print("1add mat col=",col_no,"\n ",table,"\n pivot_len",pivot_len)
 
    table.iloc[:pivot_len,col_no].fillna(start_mean,inplace=True)   #,method='ffill')
 #   print("2add mat col=",col_no,"\n ",table)

   # table.iloc[pivot_len:,col_no]=np.nan
    
   # print("3add mat col=",col_no,"\n ",table)
    return table


def filter_series_on_str(table,mask_str):
    return table.filter(like=mask_str,axis=0)


def remove_a_series_subset(table,mask_str):   #,col_name_list,window_size):
    if table.index.nlevels>1: 
        if len(table.columns)>1:
            tc = [''.join(col).strip() for col in table.columns.values]
        #    print("len>1 multi index tc=",tc)
            mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]
      #      table=table.T
     #       table=table[mask]

        else:
            tc=list(table.columns.values)
        #    print("len <=1 multi index tc=",tc)
           # mask=(mask_str not in tc)
            mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]

    #        table=table.T
    else:    
        tc=list(table.columns.values) 
      #  print("single index tc=",tc)
        mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]
      #  mask=(mask_str not in tc)
   #     table=table.T

    table=table.T
    table=table[mask]
  #  print("final table=\n",table,table.shape)
    #mask=[(mask_str not in tc[elem])  for elem in range(0,len(tc))]
 #   table=table.T
    #table=table[mask]
    return table

  

def convert_pivot_table_to_numpy_series(series_table):

#     mat_sales=series_table.iloc[row_no_list,:].to_numpy()
     mat_sales=series_table.to_numpy()

     mat_sales=np.swapaxes(mat_sales,0,1)
     mat_sales=mat_sales[np.newaxis] 
     return mat_sales



def build_mini_batch_input2(series,no_of_batches,no_of_steps):
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





def build_mini_batch_input(series,no_of_batches,batch_length):
    print("build",no_of_batches,"mini batches")
    batch_length+=1  # add an extra step which is the target (y)
    np.random.seed(45)
    # series_table= pd.read_pickle("series_table.pkl")
    # #print("series table=\n",series_table,series_table.shape)
    
    # input_length=series_table.shape[0]
    no_of_steps=series.shape[1]
    
    # print("batch_length=",batch_length)
    # print("input length",input_length)
  #  print("no of stepsh=",no_of_steps)
    
    # number_of_batches=50000
    repeats_needed=round(no_of_batches/(no_of_steps-batch_length+1),0)
    #print("rn=",repeats_needed)
    
    gridtest=np.meshgrid(np.arange(0,batch_length),np.arange(0,no_of_steps-batch_length+1))
    #start_index=gridtest[0]+gridtest[1]
    #start_index=np.repeat(start_index,repeats_needed,axis=0)   #[:,:,np.newaxis]
    start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
    
    #print("0start index=",start_index,start_index.shape)
    
    #start_index=np.repeat(start_index,input_length,axis=1)
    
    #print("1start index=",start_index,start_index.shape)
    np.random.shuffle(start_index)
     
    #print("2start index=",start_index,start_index.shape)
    #print("3start index=",start_index[:,1],start_index[:,1].shape)
    #print("4start index=",start_index[0,:],start_index[0,:].shape)
    
#    new_batches=series.to_numpy()[:,start_index]
    new_batches=series[0,start_index,:]

    #print("batches=\n",batches,batches.shape)
    #new_batches=batches[:,start_index]
 #   print("1new_batches=\n",new_batches,new_batches.shape)
   # new_batches=np.swapaxes(new_batches,0,1)
   # new_batches=np.swapaxes(new_batches,1,2)
    print("batches complete. batches shape:",new_batches.shape)
    return new_batches[:,:no_of_steps,:],new_batches[:,1:no_of_steps+1,:]







def flatten_multi_index_column_names(table):
    table=table.T

 #   print("1flatten col names table=\n",table)
 
    table.columns=table.columns.to_flat_index()

    flat_column_names = [''.join(col).strip() for col in table.columns] 
 #   print("2flatten col names table=\n",flat_column_names)
    table_col=list(table.columns)
    
    rename_dict=dict(zip(table_col, flat_column_names))
  #  print("rename dict=",rename_dict)
    #print("table_col=\n",table_col)
    table.rename(rename_dict, axis='columns',inplace=True)
  #  print("table=\n",table)
    return table.T,flat_column_names
#     del cols[-1]  # delete reference to period column at end of list
#    # print("2cols=",cols)
#    # cols_dict=dict(cols)  
#    # print("cols dict",cols_dict)
#   #  print("index shape=",np.shape(cols))
#     flat_column_names = [''.join(col).strip() for col in cols] 

#   #  fcn_dict=dict(flat_column_names)    
#   #  print("fcn dict",fcn_dict)

#     rename_dict=dict(zip(cols, flat_column_names))
#     print("rename dict",rename_dict)
# #   print("tc=",tc)
#  #   flat_column_names = [a_tuple[0][level] for a_tuple in np.shape(cols[level])[1] for level in np.shape(cols)[0]]
#   #  print("fcn=",flat_column_names)
#     table.rename(rename_dict, axis='columns',inplace=True)
 #   print("2flatten col names table=\n",table)
   
 #   return table.T



def graph_whole_pivot_table(series_table,dates): 
    series_table=series_table.T
    np.random.seed(43) 
#   print("gwpt dates",dates,len(dates))
   # series_table=series_table.T 
    plt.legend(loc="best",fontsize=8)

    cols=list(series_table.columns)
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
    ax = plt.gca()
    ax.tick_params(axis = 'x', which = 'major',rotation=45, labelsize = 8)

#    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
#    ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
 
 #   cols=flatten_multi_index_column_names(series_table)

 #   cols=list(series_table.columns)
    
  #  print("1cols=",cols)
  #  del cols[-1]  # delete reference to period column
  #  print("2cols=",cols)
  #   tc = [''.join(col).strip() for col in cols]   
  #   first_tc = [a_tuple[0] for a_tuple in cols]
  #   print("graph whole cols",cols,"\n tc",tc,"\n ftc",first_tc)
  #  colors = iter(cm.rainbow(np.linspace(0, 1, len(cols))))
    #colors = cm.rainbow(np.linspace(0, 2, len(cols)))
  #  plt.legend(loc="best",fontsize=8)
    plt.title("Sales per day",fontsize=9)
    plt.ylabel("Units or $",fontsize=9)
    plt.xlabel("Date",fontsize=8) 

    for col in cols:
        color=np.random.rand(len(cols),3)
        series_table.plot(kind='line',x='period',y=col,color=color,ax=ax,fontsize=8)

    save_fig("whole_pivot_table")
    plt.show()

  #  print("graph finished")
  #  print("graph whole cols=\n",cols)
    return cols 
 


def extend_pivot_table(series_table,dates,periods_len): 
    first_date=series_table.index[0].strftime('%Y-%m-%d')
    last_date=series_table.index[-1].strftime('%Y-%m-%d')
  #  series_table=series_table.T
   # if series_table.index.nlevels>=1:
   #     series_table.columns=series_table.columns.astype(str)
 #   product_names=series_table.columns.astype(str)
    product_names=series_table.columns   #[1:]   #.astype(str)

 #   print("xtend series table",product_names)
    len_product_names=len(product_names)
 #   print("len pn=",len_product_names)
    pidx = pd.period_range(first_date, periods=periods_len)   # 2000 days
 #   print("series_table=\n",series_table,series_table.index)
    new_table = pd.DataFrame(np.nan, index=product_names,columns=pidx)   #,dtype='category')  #series_table.columns)
    new_table=new_table.T
 #   print("new table=\n",new_table,new_table.columns)
   
 #   series_table=series_table.T
    extended_table=series_table.join(new_table,how='outer',rsuffix="_r?",sort=True) 
 #   print(" 1extend table ->extended_table=\n",extended_table,extended_table.shape)  #[:4,:4].to_string())

  #  print("range to del=",range(len_product_names,len_product_names*2))
    extended_table.drop(extended_table.columns[range(len_product_names,len_product_names*2)],axis=1,inplace=True)
  #  print(" 2extend table ->extended_table=\n",extended_table,extended_table.shape)  #[:4,:4].to_string())

    exdates=extended_table.index.astype(str).tolist()  #.astype(str)) #strftime("%Y-%m-%d"))
  #  print(" 3extend table ->extended_table=\n",extended_table,extended_table.shape)  #[:4,:4].to_string())

    return extended_table.T,exdates
 
    
    
def actual_days(series_table):
  #  print("ad=",series_table.index[0])
    first_date=series_table.index[0].to_timestamp(freq="D",how="S")
    last_date=series_table.index[-1].to_timestamp(freq="D",how="S")
    return (last_date - first_date).days +1    #.timedelta_series.dt.days    

    
  
def add_a_new_series(table,arr_names,arr,start_point,predict_ahead_steps,periods_len):
  
  #  print("ans input table shape",table,table.shape)
  #  print("add a new series first date=",table.index[0])
 #   print("ans arr_names",arr_names)
 #   print("ans arr[0]",arr[0].shape)
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
    
     
        

def find_series_type(series_name):
    return series_name[series_name.find(':')+1:]

   

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
    save_fig("log_learning_curve")
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

    predict_ahead_steps=830

 #   epochs_cnn=1
    epochs_wavenet=12
    no_of_batches=50000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
#    y_length=1
    neurons=1000  #2000
 
    pred_error_sample_size=20
    
    patience=24
    
    # dictionary mat type code :   aggsum field, name, color
    mat_type_dict=dict({"u":["qty","units","b-"]
                      #  "d":["salesval","dollars","r-"],
                      #  "m":["margin","margin","m."]
                       })
   
    mats=[14,30]   #omving average window periods for each data column to add to series table
    start_point=np.max(mats)+15  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
    mat_types=["u"]  #,"d","m"]
   
    units_per_ctn=8
   
    # train validate test split 
    train_percent=0.7
    validate_percent=0.2
    test_percent=0.1
    
    
   # filename="NAT-raw310120_no_shop_WW_Coles.xlsx"
    filename="NAT-raw240420_no_shop_WW_Coles.xlsx"
 
       #     filename="allsalestrans020218-190320.xlsx"   
    
    #index_code=['code','productgroup'] 
 #   index_code=['code','productgroup'] 
    index_code=['product'] 
  
   # index_code=['code','product'] 
   # index_code=['code']
   # index_code=['product']  
  #  index_code=['productgroup'] 
   # index_code=['code']
   # index_code=['code','productgroup','product']
###############################################33
  #  you also need to change the mask itself which is in the load data function
#################################################    
    
    
 #   mats=[30,90]   #omving average window periods for each data column to add to series table
 #   mat_types=["d"]  #,"d","m"]
  
    
    print("\nexcel input data filename='",filename,"'\n")
    print("excel query fields=",index_code)
    print("moving average days",mats)
    print("start point",start_point)
    print("predict ahead steps=",predict_ahead_steps,"\n")
    


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
        series_table,dates=load_data(filename,index_code,mat_type_dict)    #,col_name_list,window_size)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)

     #   series_table=series_table.T
     
    #    if series_table.index.nlevels>1: 
    #        series_table.index = series_table.index.map(''.join).astype(str) 
     #   series_table=series_table.T
        
     
     #   print("series table=\n",series_table,series_table.shape)
     #   print("dates=",dates,len(dates))
        actual_days_in_series_table=actual_days(series_table)
        
        
        print("actual days in series table=",actual_days_in_series_table)
        
        
        periods_len=actual_days_in_series_table+predict_ahead_steps # np.max(mats)
        print("total periods=",periods_len)

   
        
        #  this adds 

        
        series_table,extended_dates=extend_pivot_table(series_table,dates,periods_len)
    #    print("series table[start point...=\n",series_table,series_table.shape)
    #    print("extended dates=",extended_dates,len(extended_dates))
  
        series_table=add_all_mats(series_table,mats,actual_days_in_series_table,mat_types)
     #   print("series table with mats=\n",series_table,series_table.shape)
      #  print("len extended dates=",len(extended_dates))
        
        series_table,product_names=flatten_multi_index_column_names(series_table)
      #  print("series table flattened=\n",series_table,series_table.shape)
        
       
        #product_names=list(series_table.T.columns)
      #  print("product names list=",product_names)
        
        graph_whole_pivot_table(series_table.iloc[:,:actual_days_in_series_table],extended_dates[:actual_days_in_series_table])
        
        
      #  graph_whole_pivot_table(extended_series_table,extended_dates)
     #   product_names=list(series_table.index) 
   #     print("\nProduct names, length=",product_names,len(product_names))

     
        print("Saving pickled table - series_table.pkl",series_table.shape)

    #    series_table=series_table.T       
        pd.to_pickle(series_table,"series_table.pkl")
      #  print("Saving pickled extended_series_table - extended_series_table.pkl")
      #  pd.to_pickle(extended_series_table,"extended_series_table.pkl")

        series_table.T.to_csv("series_table.csv")    
        
    #    series_table=series_table.T
 
    #    print("Saving product names",product_names)
        with open("product_names.pkl","wb") as f:
            pickle.dump(product_names,f)
     #   np.save("product_names.npy",np.asarray(product_names))
        
        print("Saving dates",len(extended_dates))
        with open('dates.pkl', 'wb') as f:
            pickle.dump(dates,f)   
        with open('extended_dates.pkl', 'wb') as f:
            pickle.dump(extended_dates,f)   
 
     #   scaled_series=series_table.to_numpy()
        scaled_series=series_table.to_numpy()
        #scaled_series=scaled_series[:,start_point:actual_days_in_series_table]
        scaled_series=scaled_series[:,:actual_days_in_series_table]

    #    print("1scaled series=\n",scaled_series[:20],scaled_series.shape)
        scaled_series=np.nan_to_num(scaled_series,0)
     #   print("2scaled series=\n",scaled_series[:20],scaled_series.shape)
         
      
     #   print("scaled series=",scaled_series,scaled_series.shape)
        mat_sales_x=np.swapaxes(scaled_series,0,1)
       # mat_sales_x=np.swapaxes(scaled_series,0,1)
 
        mat_sales_x=mat_sales_x[np.newaxis] 
 
        print("Build batches")
        print("mat sales_x.shape=\n",mat_sales_x.shape)

       
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
        print("loading product_names")  #,product_names)
        with open('product_names.pkl', 'rb') as f:
             product_names = pickle.load(f)   
        print("product names=",product_names)     
    #    product_names=list(np.load("product_names.npy"))
       # # dates=list(np.load("periods.npy",allow_pickle=True))
        print("loading dates")
        with open('dates.pkl', 'rb') as f:
             dates = pickle.load(f)   
        with open('extended_dates.pkl', 'rb') as f:
             extended_dates = pickle.load(f)   
   #     print("len dates",len(dates))
   #     print("\nlen extended dates",len(extended_dates),"\n")
         
        series_table= pd.read_pickle("series_table.pkl")
        print("Loading pivot table",series_table.shape) 
      #  series_table=series_table.T
   #     extended_series_table= pd.read_pickle("extended_series_table.pkl")

       # actual_days_in_series_table=actual_days(series_table)
        
  #      print("loaded series_table=\n",series_table,series_table.shape)
      #  product_names=list(series_table.index) 
     #   print("\nProduct names, length=",product_names,len(product_names))

      #  periods_len=actual_days_in_series_table+predict_ahead_steps+1
      #  print("PERIODS=",periods_len)

   


        print("Loading mat_sales_x")
        mat_sales_x=np.load("mat_sales_x.npy")
        
        actual_days_in_series_table=actual_days(series_table.T)
        
               
#        product_names=list(series_table.index) 
#        print("\nProduct names, length=",product_names,len(product_names))

        periods_len=actual_days_in_series_table
 #       print("PERIODS=",periods_len)

   
    
           
  #  series_table=series_table.T
#    print("\n\nseries table loaded shape=",series_table.shape,"\n")  
#    print("product names=\n",product_names)
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
    print("batch_length=",batch_length)
    print("n_inputs=",n_inputs)
    print("predict_ahead_steps=",predict_ahead_steps)
    print("full prediction day length=",periods_len)

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
        
        print("\n Neurons=",neurons,"[wavenet]. Building and compiling model\n")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        layer_count=1    

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[None,n_query_rows]))
        model.add(keras.layers.BatchNormalization())
        for rate in (1,2,4,8) *2:
            
            model.add(keras.layers.Conv1D(filters=neurons, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate))
 
#            if (n_inputs>2) & ((layer_count==3) | (layer_count==5)):  # | (layer_count==5) | (layer_count==6):
            if ((layer_count==3) | (layer_count==5)):  # | (layer_count==4) | (layer_count==1) | (layer_count==6) | (layer_count==7)):

                #     model.add(keras.layers.Dropout(rate=0.2))   # calls the MCDropout class defined earlier
                model.add(keras.layers.BatchNormalization())
 
            if ((layer_count==2) | (layer_count==4)):   
             #    model.add(keras.layers.BatchNormalization())
                 model.add(keras.layers.AlphaDropout(rate=0.2))   # calls the MCDropout class defined earlier            
              #   model.add(keras.layers.MCDropout(rate=0.2))   # calls the MCDropout class defined earlier
        #    model.add(keras.layers.Conv1D(filters=neurons, kernel_size=2,padding='causal',activation='relu',dilation_rate=rate))
 
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
    print("series=\n",list(series_table.index),"\n")
    print("Loading mat_sales_x")
    mat_sales_x=np.load("mat_sales_x.npy")

    model=keras.models.load_model("wavenet_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})

    print("mat_sales_x shape",mat_sales_x.shape)
   

    original_steps=mat_sales_x.shape[1]
    ys= model.predict(mat_sales_x[:,start_point:,:])    #[:, np.newaxis,:]
    print("ys shape",ys.shape)

  #  pas=predict_ahead_steps   #,dtype=tf.none)
    
    for step_ahead in range(1,predict_ahead_steps):
          print("\rstep:",step_ahead,"/",predict_ahead_steps,"ys shape",ys.shape,end='\r',flush=True)
          y_pred_one = model.predict(ys[:,:(start_point+step_ahead),:])[:, np.newaxis,:]  #[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
          ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1)    #[:, np.newaxis,:]), axis=1)
     #     print("ys",ys,"ys shape",ys.shape,step_ahead)
         

    print("\rstep:",step_ahead+1,"/",predict_ahead_steps,end='\n\n',flush=True)
    pred_product_names=[s + "pred" for s in original_product_names]    #original_product_names
   #  print("pred product names=\n",pred_product_names)
#    print("est before=",extended_series_table,extended_series_table.shape)  
#
#   Pad ys array with nans to fit it easily into the extended table.  Add startpoint
  #  print("ys before start padding=\n",ys.shape)
 #   pad_before_arr = np.empty((1,start_point+1,ys.shape[2]))
 #   pad_before_arr[:] = np.NaN
  #  print("pad before arr=\n",pad_before_arr.shape)

#    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
 #   ys = np.concatenate((pad_before_arr,ys),axis=1) 
 #   print("ys after the before padding=\n",ys.shape)

 #   print("periods_len=",periods_len,"start point",start_point)
 #   print("ys before end padding=\n",ys,ys.shape)
   # if periods_len>(ys.shape[1]-start_point):
#     pad_after_arr = np.empty((1,periods_len-ys.shape[1],ys.shape[2]))
#     pad_after_arr[:] = np.NaN
#     print("pad after arr=\n",,pad_after_arr.shape)

# #    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
#     ys = np.concatenate((ys,pad_after_arr),axis=1) 
#    # else:
#     #    print("periods len <= ys.shape. ys=\n",ys,ys.shape)
#     #    ys=ys[:,:(periods_len+start_point),:]
        
 #   print("ys after end  padding=\n",ys.shape)



    series_table,product_name,extended_dates=add_a_new_series(series_table,pred_product_names,ys,start_point,predict_ahead_steps,periods_len)
 
  #  print("\n 1series table=\n",series_table.T.columns,series_table.shape)
   
 #  print("est after=",series_table,series_table.shape,product_names)  

  #  print("est dates=",extended_dates,len(extended_dates))

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
     
     
   # pas=predict_ahead_steps   #,dtype=tf.none)
     
    for step_ahead in range(1,predict_ahead_steps):
       print("\rstep:",step_ahead,"/",predict_ahead_steps,"mc_ys shape",mc_ys.shape,end='\r',flush=True)
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
     
     
    print("\rstep:",step_ahead+1,"/",predict_ahead_steps,end='\n\n',flush=True)
    
    
    #########################################33 
       
    pred_product_names=[s + "pred_mc" for s in original_product_names]
     
         #   Pad mc_ys array with nans to fit it easily into the extended table.  Add startpoint
             #  print("mc_ys before padding=\n",mc_ys,mc_ys.shape)
 #   pad_before_arr = np.empty((1,start_point+1,mc_ys.shape[2]))
 #   pad_before_arr[:] = np.NaN
     #  print("pad before arr=\n",pad_before_arr,pad_before_arr.shape)
     
         #    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
  #  mc_ys = np.concatenate((pad_before_arr,mc_ys),axis=1) 
     #  print("mc_ys after the before padding=\n",mc_ys,mc_ys.shape)
     
            # if periods_len>mc_ys.shape[1]:
 #   pad_after_arr = np.empty((1,periods_len-mc_ys.shape[1],mc_ys.shape[2]))
 #   pad_after_arr[:] = np.NaN
     
    
             #  print("pad after arr=\n",pad_after_arr,pad_after_arr.shape)
     
             #    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
  # mc_ys = np.concatenate((mc_ys,pad_after_arr),axis=1) 
         # else:
                 #    mc_ys=mc_ys[:,:(periods_len+start_point),:]
     
         #  print("mc_ys after end  padding=\n",mc_ys,mc_ys.shape)
     
               #  series_table=series_table.T
    series_table,product_names,extended_dates=add_a_new_series(series_table,pred_product_names,mc_ys,start_point,predict_ahead_steps,periods_len)
    
  #  print("\n 2series table=\n",series_table.columns,series_table.shape)
   
     #############################################################
     
    pred_product_names=[s + "yerr_mc" for s in original_product_names]  
     
     
             #   Pad mc_yerr array with nans to fit it easily into the extended table.  Add startpoint
                         #  print("mc_yerr before padding=\n",mc_yerr,mc_yerr.shape)
   # pad_before_arr = np.empty((1,start_point+1,mc_yerr.shape[2]))
   # pad_before_arr[:] = np.NaN
             #  print("pad before arr=\n",pad_before_arr,pad_before_arr.shape)
     
         #    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
 #   mc_yerr = np.concatenate((pad_before_arr,mc_yerr),axis=1) 
             #  print("mc_yerr after the before padding=\n",mc_yerr,mc_yerr.shape)
     
           #  if periods_len>mc_yerr.shape[1]:
  #  pad_after_arr = np.empty((1,periods_len-mc_yerr.shape[1],mc_yerr.shape[2]))
  #  pad_after_arr[:] = np.NaN
         #  print("pad after arr=\n",pad_after_arr,pad_after_arr.shape)
     
         #    ys = np.concatenate((ys,y_pred_one[:,:,-1,:]),axis=1) 
  #  mc_yerr = np.concatenate((mc_yerr,pad_after_arr),axis=1) 
             #  print("mc_yerr after end  padding=\n",mc_yerr,mc_yerr.shape)
                     #else:
                                 #    mc_yerr=mc_yerr[:,:(periods_len+start_point),:]
    
     
              #   series_table=series_table.T    
    series_table,product_names,extended_dates=add_a_new_series(series_table,pred_product_names,mc_yerr,start_point,predict_ahead_steps,periods_len)
    
    
    print("MC dropout predict finished\n")
    
    
 #   print("\n 3series table=\n",series_table.columns,series_table.shape)
##############################################################
  #  print("product names=",product_names)
 #   series_table=series_table.T
    for p in range(0,series_table.shape[0]):
    #    plt.figure(figsize=(11,4))
        plt.figure(figsize=(14,5))
 
        plt.subplot(121)
        
        ax = plt.gca()
    #    ax.tick_params(axis = 'both', which = 'major',rotation=90, labelsize = 6)
        ax.tick_params(axis = 'x', which = 'major',rotation=45, labelsize = 8)

    #       ax.tick_params(axis = 'both', which = 'minor', rotation=90, labelsize = 6)

        plt.title("Sales/day:Actual+Pred: "+str(product_names[p]),fontsize=10)
      
        plt.ylabel("Units or $ / day",fontsize=9)
        plt.xlabel("Date",fontsize=9) 
        graph_a_series(series_table,extended_dates,str(product_names[p]))
        #,series_table.columns)
        
        plt.legend(loc="best",fontsize=8)
        save_fig("sales_pred_"+str(p))
        plt.show()
     
        
     
   # print("Save graph")
   # save_fig("sales_pred")


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
    
    print("\nwrite predictions to sales_prediction.CSV file....")
#    dates.sort()
  
   # print("extended series table=\n",extended_series_table)
    
  #  print("\nseries table shape=",series_table.shape)

  #  print("First date=",extended_dates[0])
  #  print("last date=",extended_dates[-1])
    with open("sales_prediction_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
        series_table.T.to_csv(f)  #,line_terminator='rn')
    

  
###############################################################

    print("Saving pickled final table - final_series_table.pkl",series_table.shape)

    #    series_table=series_table.T       
    pd.to_pickle(series_table,"final_series_table.pkl")
   
    with open("calc_sales_prediction_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
        series_table.to_csv(f)  #,line_terminator='rn')
    
    
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
    forecast_table = st.resample('W', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
    
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
    #test=teste.set_levels([['a', 'b'], [1, 2]],  level=[0, 1])
  #  print("test",test,test.shape)
  
    
    # with open("calc_sales_prediction_"+str(product_names[0])+".csv", 'w') as f:  #csvfile:
    #     forecast_table.to_csv(f)  #,line_terminator='rn')
 
    
    
#   #  now = dt.datetime.now()   #.strftime('%d/%m/%Y')
#  #   now_dt = datetime.strptime(now, '%Y/%m/%d').date()
#   #  print("now=",now)
#     dbd2["new_date"]=start_date  #.dt.date
#     dbd2['new_date'] = dbd2['new_date'] + pd.to_timedelta(dbd2['day_order_delta'], unit='d')
#     if cfg.dateformat=="year/week":
#         dbd2['predict_date']=dbd2['new_date'].dt.strftime('%Y/%w')    # ('%Y/%m/%d")
#     elif cfg.dateformat=="year/month/day":
#         dbd2['predict_date']=dbd2['new_date'].dt.strftime('%Y/%m/%d')
#     elif cfg.dateformat=="year/month":
#         dbd2['predict_date']=dbd2['new_date'].dt.strftime('%Y/%m')
        
#    # new_datetime_obj = datetime.strptime(orig_datetime_obj.strftime('%d-%m-%y'), '%d-%m-%y').date()
 
#     dbd2= dbd2[["code","product","predict_date","predict_qty"]]
    
#     dbd2.sort_values(by=["predict_date"],axis=0,ascending=[False],inplace=True)
#     dbd2["predict_qty"]=dbd2["predict_qty"].astype(float).round(0)    #{"predict_qty" : 1})
#     dbd2["predict_qty_ctnsof8"]=(dbd2["predict_qty"]/8).astype(float).round(0)
#     #   print(k)
#     #print("Sales Qty Predictions=\n",dbd2[0:100].to_string())
    
   
# #############################################################################3
#     #  create a pivot table of code, product, day delta and predicted qty and export back to excel

#     table = pd.pivot_table(dbd2, values='predict_qty', index=['product', 'predict_date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)   #, observed=True)
#   #  print("\ntable=\n",table.head(5))
#     f.write("\n\n"+table.to_string())

#     table2 = pd.pivot_table(dbd2, values='predict_qty', index=['code', 'predict_date'],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
#   #  print("\ntable2=\n",table2.head(5))
#     f.write("\n\n"+table2.to_string())

#     table3 = pd.pivot_table(dbd2, values='predict_qty', index=['predict_date',"code"],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
#   #  print("\ntable3=\n",table3.head(5))
#     f.write("\n\n"+table3.to_string())

#     table4 = pd.pivot_table(dbd2, values='predict_qty', index=['predict_date',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
#   #  print("\ntable4=\n",table4.head(5))
#     f.write("\n\n"+table4.to_string())

#     table5 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['product', 'predict_date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
#   #  print("\ntable5=\n",table5.head(5))
#     f.write("\n\n"+table5.to_string())

#     table6 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['code', 'predict_date'],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
#   #  print("\ntable6=\n",table6.head(5))
#     f.write("\n\n"+table6.to_string())

#     table7 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['predict_date',"code"],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
#   #  print("\ntable7=\n",table7.head(5))
#     f.write("\n\n"+table7.to_string())

#     table8 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['predict_date',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
#   #  print("\ntable8=\n",table8.head(5))
#     f.write("\n\n"+table8.to_string())
    s=str(new_col_names[0])
    s = s.replace(',', '_')
    s = s.replace("'", "")
    s = s.replace(" ", "")

    with pd.ExcelWriter("SCB_"+s+".xls") as writer:  # mode="a" for append
        forecast_table.to_excel(writer,sheet_name="Units1")
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


    
          
          
          
