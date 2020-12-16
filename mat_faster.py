#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:06:46 2020

@author: tonedogga
"""

import pandas as pd
import glob
import imageio
#import cv2
import shutil
from p_tqdm import p_map,p_umap
from pathlib import Path
 
import datetime as dt
from datetime import datetime,timedelta
import dash2_dict as dd2


def get_rolling_amount(grp, freq):
     return grp.rolling(freq,on=grp.index)['salesval'].sum()


def mat_salesval_faster(slices):
     t_df=slices["df"]    
     k=slices['name']
     start=slices['start']   #+pd.offsets.Day(365)
   #  start_2yrs=slices['start']
     end=slices['end']
     latest_date=slices['end']
     output_dir=slices["plot_dump_dir"]
  #   print("slices=",slices)
  #   print("se",start,end)
     #plot_vals=[]
  #   for d in pd.date_range(start,end):
      
     mat_df['mat'] = sales_df.groupby('salesval', as_index=False, group_keys=False).apply(get_rolling_amount, '365D')
      
   #  mat_df=t_df[(t_df.index>=(start+pd.offsets.Day(-365))) & (t_df.index<end)]
 
       #  print("v=\n",v) 
      #   mat= 28 #dd2.dash2_dict['sales']['plots']['mat']
         
         
                #   mat_df=self.preprocess(mat_df,mat)
    
  #   mat_df=v.resample('D',label='left').sum().round(0).copy()
         
      #   print("s",s)   #,s.iloc[-1])
  #   mat_df['mat']=mat_df['salesval'].rolling(365,axis=0).sum()
   #  print("plotvals=",plot_vals)
         
 
     #plot_df = pd.DataFrame(plot_vals, columns =['date', 'salesval'])
     #plot_df.set_index('date',inplace=True)
   #  print("plotdf=\n",plot_df)
    # mat_df=t_df[(u_df.index>=start) & (u_df.index<end)].copy()
     #if mat_df.shape[0]>367:
      #   mat_df['mat']=mat_df['salesval'].rolling(365,axis=0).sum()
      #   print("resampled mat_df=\n",mat_df)
 
           #      df=df[(df['mat']>=0)]
     print(mat_df) 
     #     print("end mat preprocess=\n",df)
         # styles1 = ['b-','g:','r:']
     styles1 = ['b-']
     # styles1 = ['bs-','ro:','y^-']
     linewidths = 1  # [2, 1, 4]
              
    #fig, ax = pyplot.subplots()
     mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
 
 

def load(save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            return pd.read_pickle(save_dir+savefile)
        else:
            print("load sales_df error.")
            return
 

 
def generate_annual_dates(df,*,start_offset,size):
     first_date=df.index[0]+pd.offsets.Day(size+start_offset)
     last_date=df.index[-1]  #first_date+pd.offsets.Day(365)
 
    # start_date=pd.to_datetime(start_date)
    # end_date=pd.to_datetime(end_date)
     
     dr=[d for d in pd.date_range(first_date,last_date)]
     for date in dr:
         yield date+pd.offsets.Day(-size),date
         
  
    
  
    
sales_df=load("./","sales_trans_df.pkl")
sales_df.drop(['date'],axis=1,inplace=True)

#print(sales_df.columns,sales_df.head())
#mat_df=sales_df.groupby('date', as_index=True, sort=False,group_keys=True).apply(get_rolling_amount,'365D').to_frame(name='mat')
mat_df=sales_df.groupby('date').agg("sum")[['salesval']]
#print('1',mat_df)
mat_df['mat']=mat_df['salesval'].rolling("365D",closed='left').sum()   #   apply(get_rolling_amount,'365D')   #.to_frame(name='mat')    #.rolling("365D", min_periods=1).sum()

print(mat_df)
#mat_df=mat_df.droplevel(1,axis=0)
#print(mat_df)
#m_df=mat_df.to_frame(name='mat')
#mat_df.sort_index(inplace=True)
#print(mat_df.tail(100).to_string())
#m_df.columns(1).name='mat'
#m_df.set_index('date',inplace=True)

#print(m_df)
for start,end in generate_annual_dates(mat_df,start_offset=365,size=365):  #query_dict['080 SA all']):  #"2020-01-01","2020-02-02"):
    print("mats",start,"to",end)
#    slices.append({"start":start,"end":end,"plot_dump_dir":".","name":key+" ["+start.strftime("%Y-%m-%d")+"-"+end.strftime("%Y-%m-%d")+"]","df":sales_df[(sales_df.index>start) & (sales_df.index<=end)]})  
    plot_df=mat_df[(mat_df.index>start) & (mat_df.index<=end)].copy()
    plot_df['mat'].plot(use_index=True)
    
#mat_df['mat'].plot(use_index=True)
# slices=[] 
# key="key"
# for start,end in generate_annual_dates(sales_df,start_offset=365,size=365):  #query_dict['080 SA all']):  #"2020-01-01","2020-02-02"):
#    print("mats",start,"to",end)
#    slices.append({"start":start,"end":end,"plot_dump_dir":".","name":key+" ["+start.strftime("%Y-%m-%d")+"-"+end.strftime("%Y-%m-%d")+"]","df":sales_df[(sales_df.index>start) & (sales_df.index<=end)]})  

# print(slices)
#print(mat_df) 