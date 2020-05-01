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
#import SCBS0 as c

filename="tables_dict.pkl"


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm



# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# from tensorflow import keras
# assert tf.__version__ >= "2.0"

# print("\n\nTurn a list of tables into a list of batches - By Anthony Paech 25/2/20")
# print("========================================================================")       

# print("Python version:",sys.version)
# print("\ntensorflow:",tf.__version__)
# print("keras:",keras.__version__)
# print("sklearn:",sklearn.__version__)
import os

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)




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

#n_steps = 50

def save_fig(fig_id, images_path, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)






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



# def build_mini_batch_input2(series,no_of_batches,no_of_steps):
#     print("build mini batch series shape",series.shape)
#     np.random.seed(42) 
#     series_steps_size=series.shape[1]

#     random_offsets=np.random.randint(0,series_steps_size-no_of_steps,size=(no_of_batches)).tolist()
#     new_mini_batch=series[:,random_offsets[0]:random_offsets[0]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)

#    # prediction=new_mini_batch[:,-1]
#     for i in range(1,no_of_batches):
#         temp=new_mini_batch[:,:no_of_steps+1]
#         new_mini_batch=series[:,random_offsets[i]:random_offsets[i]+no_of_steps+1]    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)
#     #    prediction=np.concatenate((prediction,new_mini_batch[:,-1]))
#         new_mini_batch=np.vstack((temp,new_mini_batch[:,:no_of_steps+1]))
        
#         if i%100==0:
#             print("\rBatch:",i,"new_mini_batch.shape:",new_mini_batch.shape,flush=True,end="\r")
#     return new_mini_batch[:,:no_of_steps,:],new_mini_batch[:,1:no_of_steps+1,:]





def build_mini_batch_input(series,no_of_batches,batch_length):
    print("build",no_of_batches,"mini batches")
  #  batch_length+=1  # add an extra step which is the target (y)
    np.random.seed(45)
    no_of_steps=series.shape[1]
    
    print("batch_length=",batch_length)
    print("no of steps=",no_of_steps)
 
    repeats_needed=round(no_of_batches/(no_of_steps-batch_length),0)
    
    gridtest=np.meshgrid(np.arange(0,batch_length+1),np.arange(0,no_of_steps-(batch_length+1)))
    start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
    np.random.shuffle(start_index)
    new_batches=series[0,start_index,:]
    np.random.shuffle(new_batches)
    print("batches complete. batches shape:",new_batches.shape)
    return new_batches[:,:batch_length,:],new_batches[:,1:batch_length+1,:]







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



def graph_whole_pivot_table(series_table,dates,query_name,images_path): 
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
    plt.title(str(query_name)+" Sales per day",fontsize=9)
    plt.ylabel("'u'=Units, 'd'=$",fontsize=9)
    plt.xlabel("Date",fontsize=8) 

    for col in cols:
        color=np.random.rand(len(cols),3)
        series_table.plot(kind='line',x='period',y=col,color=color,ax=ax,fontsize=8)

    save_fig("whole_pivot_table_"+str(query_name),images_path)
    plt.show()

  #  print("graph finished")
  #  print("graph whole cols=\n",cols)
    return cols 
 


def extend_pivot_table(series_table,periods_len,actual_days_in_series_table,required_starting_length,predict_ahead_steps,start_point): 
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

 #   print("load extended table before '",extended_table.shape,"'")
    if actual_days_in_series_table>=required_starting_length:
         print("current start point=",start_point)
         print("extra start point needed=",actual_days_in_series_table-required_starting_length)
    #     extended_table=extended_table.iloc[:,-(required_starting_length+predict_ahead_steps):] 
         print("current start point=",start_point)
        # start_point=start_point+(actual_days_in_series_table-required_starting_length)           
         actual_days_in_series_table=required_starting_length
         print("start point adjusted to",start_point)
     #    print("truncated series_table to the last",required_starting_length,"+ predict ahead steps",predict_ahead_steps,"days")
    else:
         print("\nWarning... the number of days loaded (",actual_days_in_series_table,") is not enough for a synced 2 year analysis. requires (",required_starting_length,") days.\n")
#    print("load extended table after",extended_table.shape)



    exdates=extended_table.index.astype(str).tolist()  #.astype(str)) #strftime("%Y-%m-%d"))
  #  print(" 3extend table ->extended_table=\n",extended_table,extended_table.shape)  #[:4,:4].to_string())

    return extended_table.T,exdates,actual_days_in_series_table,start_point
 
    
    
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






def main(c):    
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
    
    
    print("numpy:",np.__version__)
    print("pandas:",pd.__version__)
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
    
    
    

    predict_ahead_steps=c.predict_ahead_steps
    
     #   epochs_cnn=1
    #epochs_wavenet=36
    no_of_batches=c.no_of_batches   #100000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=c.batch_length #16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
    #    y_length=1
    #neurons=1600  #1000-2000
     
    #pred_error_sample_size=40
    
    #patience=6   #5
    
    # dictionary mat type code :   aggsum field, name, color
       # mat_type_dict=dict({"u":["qty","units","b-"]
                      #  "d":["salesval","dollars","r-"],
                      #  "m":["margin","margin","m."]
    #                   })
       
    mats=c.mats   #[14]   #omving average window periods for each data column to add to series table
    start_point=c.start_point  #np.max(mats)+15  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
    mat_types=c.mat_types    #["u"]  #,"d","m"]
       
    units_per_ctn=c.units_per_ctn  #8
       
    # train validate test split 
    train_percent=c.train_percent   #0.7
    validate_percent=c.validate_percent  #0.2
    test_percent=c.test_percent  #0.1
     
    
    print("moving average days",mats)
    print("start point",start_point)
    print("predict ahead steps=",predict_ahead_steps,"\n")
    required_starting_length=c.required_starting_length    #=731+np.max(mats)+batch_length   # 2 years plus the MAT data lost at the start + batchlength
    
    
    print("\nBatch creator\n\n")
    print("unpickling '",filename,"'")  
    with open(filename, "rb") as f:
        all_tables = pickle.load(f)
      #  testout2 = pickle.load(f)
    qnames=[all_tables[k][0] for k in all_tables.keys()]    
    print("unpickled",len(all_tables),"tables (",qnames,")")
    
    
    ########################################
    
    
    
    batch_list = defaultdict(list)
    
      
    
    
    table_number=0
    for table_number in all_tables.keys():
         print("\n processing table:",qnames[table_number],"....\n")
         series_table=all_tables[table_number][1]
    
    
    
         actual_days_in_series_table=actual_days(series_table)
         
         
         print("actual days in series table=",actual_days_in_series_table)
         print("required minimum starting days for 2 year series analysis:",required_starting_length)
         
         
         periods_len=actual_days_in_series_table+predict_ahead_steps # np.max(mats)
         print("total periods=",periods_len)
        
    
         
         series_table,extended_dates,actual_days_in_series,start_point=extend_pivot_table(series_table,periods_len,actual_days_in_series_table,required_starting_length,predict_ahead_steps,start_point)
           
         series_table=add_all_mats(series_table,mats,actual_days_in_series_table,mat_types)
         
         series_table,product_names=flatten_multi_index_column_names(series_table)
          
         saved_series_table=series_table.copy(deep=True)
         print("saved series table shape=",saved_series_table.shape)
            
         graph_whole_pivot_table(series_table.iloc[:,:actual_days_in_series_table],extended_dates[:actual_days_in_series_table],qnames[table_number],c.images_path)
       
     #    series_table.T.to_csv("series_table_"+str(qnames[table_number])+".csv")    
         
         #    series_table=series_table.T
         
    
         
         #    print("Saving product names",product_names)
         with open("product_names_"+str(qnames[table_number])+".pkl","wb") as f:
             pickle.dump(product_names,f)
          #   np.save("product_names.npy",np.asarray(product_names))
         
         print("Saving dates",len(extended_dates))
         #with open('dates.pkl', 'wb') as f:
         #    pickle.dump(dates,f,protocol=4)   
         with open("extended_dates_"+str(qnames[table_number])+".pkl", 'wb') as f:
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
        
        
         # print("\n\nSave batches")
         # np.save("batch_train_X.npy",X)
         # np.save("batch_train_y.npy",y)
         
         # print("Saving mat_sales_x")
         # np.save("mat_sales_x.npy",mat_sales_x)
         
    
            
        
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
         #print("epochs_wavenet=",epochs_wavenet)
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
           
         batch_list[table_number].append(qnames[table_number])
         batch_list[table_number].append(X_train)
         batch_list[table_number].append(y_train)
         batch_list[table_number].append(X_valid)
         batch_list[table_number].append(y_valid)
         batch_list[table_number].append(X_test)
         batch_list[table_number].append(y_test)
         batch_list[table_number].append(mat_sales_x)
         batch_list[table_number].append(product_names)
         batch_list[table_number].append(saved_series_table)
     
         print("mat_sales_x shape2=",mat_sales_x.shape)
         
         print("saved series table shape2=",saved_series_table.shape)
     
     
     
         table_number+=1
         #########################################################
     
    print("final batch list len=",len(batch_list))    
    
      
    batch_dict = dict((k, tuple(v)) for k, v in batch_list.items())  #.iteritems())
    
    
    #print("\n table dict=\n",table_dict)
    
    with open("batch_dict.pkl","wb") as f:
        pickle.dump(batch_dict, f,protocol=-1)
        
    #querynames=[table_dict[k][0] for k in table_dict.keys()]    
    print("pickling",len(batch_dict))   #," (",[table_dict[k][0] for k in table_dict.keys()],")")
    
    
    #######################################################
    
    
    
    # #print(batch_dict)
    # print("\n\ntest unpickling")  
    # with open("batch_dict.pkl", "rb") as f:
    #     testout1 = pickle.load(f)
    #   #  testout2 = pickle.load(f)
    # qnames=[testout1[k][0] for k in testout1.keys()]    
    # print("unpickled",len(testout1),"tables (",qnames,")")
    
    # for n in range(len(qnames)):    
    #     print(testout1[n][0],"=\n",testout1[n][1:7]) 
    
    return


if __name__ == '__main__':
    main()

