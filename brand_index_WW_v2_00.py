#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:52:36 2020

@author: tonedogga
"""
import numpy as np
import pandas as pd
import datetime as dt
from pandas.plotting import scatter_matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


wwjamsxls="Ww_jams_scan_data_300520.xlsx"



#         self.start_point=0
        
 
def add_dates_back(df,all_dates):
    return pd.concat((df,all_dates_df),axis=1)   #,on='ww_scan_week',how='right')

       
        
        
#column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
 
col_dict=dict({0:"WW_scan_week",
               1:"BB_off_promo_sales",
               2:"BB_on_promo_sales",
               3:"SD_off_promo_sales",
               4:"SD_on_promo_sales",
               5:"BM_off_promo_sales",
               6:"BM_on_promo_sales"})
       
df=pd.read_excel(wwjamsxls,-1,skiprows=[0,1,2]).T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
#print("before",df)
df = df.rename(col_dict,axis='index').T

df['WW_scan_week']=pd.to_datetime(df['WW_scan_week'],format="%d/%m/%y")
#df['coles_scan_week']=df["date"] #.strftime("%Y-%m-%d")
df.fillna(0.0,inplace=True)
df.drop_duplicates(keep='first', inplace=True)
#df.replace(0.0, np.nan, inplace=True)
#print("after",df)

df=df.sort_values(by=['WW_scan_week'], ascending=True)
df=df.set_index('WW_scan_week') 
df=df.astype(np.float32)  #,inplace=True)
df['weekno']= np.arange(len(df))
print("final",df,df.T)

df['BB_on_promo']=(df['BB_on_promo_sales']>0.0)
df['SD_on_promo']=(df['SD_on_promo_sales']>0.0)
df['BM_on_promo']=(df['BM_on_promo_sales']>0.0)

df['BB_total_sales']=df['BB_off_promo_sales']+df['BB_on_promo_sales']
df['SD_total_sales']=df['SD_off_promo_sales']+df['SD_on_promo_sales']
df['BM_total_sales']=df['BM_off_promo_sales']+df['BM_on_promo_sales']



df.replace(0.0, np.nan, inplace=True)

#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')  #,fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BB_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='BB_on_promo')

sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')

