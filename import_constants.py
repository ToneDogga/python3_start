#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 07:17:37 2020

@author: tonedogga
"""


#  constants for TF2_ sales prediction


import numpy as np
import pandas as pd
from natsort import natsorted

import os


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




predict_ahead_steps=36
epochs_cnn=1
epochs_wavenet=25
no_of_batches=50000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
batch_length=16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
y_length=1
neurons=514
start_point=150
pred_error_sample_size=60

series_dict=dict({"mt":"blue",
                  "pred":"red",
                  "mt_pred":"red",
                  "pred_mc":"green",
                  "mt_pred_mc":"green",
                  "yerr_mc":"magenta",
                  "mt_yerr_mc":"magenta"
                  })



#predict_start_point=20
 #   plot_y_extra=1000   # extra y to graph plot 
 #   mat_length=20
#    overlap=0  # the over lap between the X series and the target y

kernel_size=4   # for CNN
strides=2   # for CNN


# train validate test split 
train_percent=0.7
validate_percent=0.2
test_percent=0.1


filename="NAT-raw310120all.xlsx"
   #     filename="allsalestrans020218-190320.xlsx"   

mats=[40]    # 22 work days is approx one month 16 series moving average window periods for each data column to add to series table


    
def load_data(mats,filename,series_dict):    #,col_name_list,window_size):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows
   
    df.fillna(0,inplace=True)
    
  #  print(df['productgroup'])
    df['productgroup']=(df['productgroup']).astype(int)
  #  print(df)

    
 #   df["period"]=df.date.dt.to_period('W')
   
    df["period"]=df.date.dt.to_period('D')
   # df["period"]=df.date.dt.to_period('B')

    
    
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
    
 #   mask=((df['code']=='FLPAS')) & (df['product']=='SJ300'))
    mask=((df['productgroup']>=10) & (df['productgroup']<=14))
 #   mask=(df['product']=='SJ300')
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
  #  dates=list(set(list(df['date'].dt.strftime("%Y-%W"))))

    dates.sort()
    
    df['period'] = df['period'].astype('category')
   # df[mask]=df[mask].astype('category')

#    table = pd.pivot_table(df[mask], values='qty', index=['productgroup'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True
    table = pd.pivot_table(df[mask], values='qty', index=['productgroup'],columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T  #observed=True

    colnames=list(table.columns)
    print("colnames=\n",colnames)
    for window_length in range(0,len(mats)):
        col_no=0
        for col in colnames:
            table.rename(columns={col: str(col)+"@1"},inplace=True)
            table=add_mat(table,col_no,col,mats[window_length],series_dict)
            col_no+=1
  
    table = table.reindex(natsorted(table.columns), axis=1) 
    if len(mats)>0:
        table=remove_some_table_columns(table,mats).T   #,col_name_list,window_size) 
  
    return table,dates




def add_mat(table,col_no,col_name,window_period,series_dict):
   #   print("table iloc[:window period]",table.iloc[:,:window_period])     #shape[1])
  #  start_mean=table.iloc[:window_period,0].mean(axis=0) 
  #  print("series dict=",series_dict["_mt",0])
    start_mean=table.iloc[:window_period,col_no].mean(axis=0) 
    
#  print("start mean=",window_period,start_mean)   # axis =1
    mat_label=str(col_name)+"@"+str(window_period)+":mt" 
    table[mat_label]= table.iloc[:,col_no].rolling(window=window_period,axis=0).mean()
    
 #   table=table.iloc[:,0].fillna(start_mean)
    return table.fillna(start_mean)




def remove_some_table_columns(table,window_size):   #,col_name_list,window_size):
    # col name is a str
    # returns anumpy array of the 
    col_filter="(@"+str(window_size)+")"
    print("col filter=\n",col_filter)
    return table.filter(regex=col_filter)

    
def add_a_new_series(table,arr_names,arr):
    table=table.T
 
    start_arr = np.empty((table.shape[0]-arr.shape[1],arr.shape[2]))
  #  print("startarr1 shape=",start_arr.shape)
    start_arr[:] = np.NaN
  #  print("start arr shape2=",start_arr.shape)
    new_arr=np.vstack((start_arr,arr[0]))  
    new_cols=pd.DataFrame(new_arr,columns=arr_names)
    new_cols['period']=table.index
    new_cols.set_index('period', inplace=True)
 #   new_cols=new_cols.T
 #   new_cols['series_type']=arr_type
  #  new_cols=new_cols.T
    
    table2=table.join(new_cols,on='period',how='left')
    table2 = table2.reindex(natsorted(table2.columns), axis=1) 
    new_product_names=table2.columns
    
    table2=table2.T
#    print("table2 cols=",table2.columns)
    print("new series added. extended_series_table=\n",table2,table2.shape) 
    print("new product names=\n",new_product_names)
    return table2,new_product_names
    

