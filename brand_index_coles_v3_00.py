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


colesjamsxls="Coles_jams_scan_data_300520.xlsx"

mats=7

#         self.start_point=0
        
 
def add_dates_back(df,all_dates):
    return pd.concat((df,all_dates_df),axis=1)   #,on='ww_scan_week',how='right')


def plot_query(query_df_passed,plot_col,query_name):
    query_df=query_df_passed.copy(deep=True)
  #  query_details=str(query_df.columns[0])
  #  query_df.columns = query_df.columns.get_level_values(0)
    print("query df shae",query_df.shape)
 #   query_df=self.mat_add_1d(query_df.to_numpy().swapaxes(0,1),self.mats[0])
    query_df['weeks']=query_df.index.copy(deep=True)  #.to_timestamp(freq="D",how='s') #
#        query_df['qdate']=pd.to_datetime(pd.Series(query_list).to_timestamp(freq="D",how='s'), format='%Y/%m/%d')
   # print("query list",query_list)
  #  query_df['qdate'].apply(lambda x : x.to_timestamp())
#    query_df['qdate']=query_list.to_timestamp(freq="D",how='s')
    query_list=query_df['weeks'].tolist()
  #  print("qudf=\n",query_df,query_df.columns[1][0])
#    print("f",query_list)
    #   query_df['qdate'] = query_df.qdate.tolist()

    for col in plot_col:
        new_query_df=query_df[col].copy(deep=True)
     #   print("1query_df=",new_query_df)
        new_query_df=new_query_df.rolling(mats,axis=0).mean()
     #   print("2query_df=",new_query_df,new_query_df.shape)
     #   fill_val=new_query_df.iloc[mats+1]  #.to_numpy()
    #    print("fill val",fill_val)
      #  print("3query_df=",new_query_df,new_query_df.shape)
  
     #   new_query_df=new_query_df.fillna(fill_val)
       # query_df.reset_index()   #['qdate']).sort_index()
     #   query_df.reset_index(level='specialpricecat')
   #     query_df.reset_index(drop=True, inplace=True)
      #  new_query_df['weeks']=query_list   #.set_index(['qdate',''])
      #  print("3query df=\n",new_query_df,col)
      #  query_df=query_df.replace(0, np.nan)
       #     ax=query_df.plot(y=query_df.columns[0],style="b-")   # actual
     #   ax=query_df.plot(x=query_df.columns[1][0],style="b-")   #,use_index=False)   # actual
        ax=new_query_df.plot(x='weeks',y=col)   #,style="b-")   #,use_index=False)   # actual

  #  col_no=1
 #   query.plot(style='b-')
 #   ax.axvline(pd.to_datetime(start_date), color='k', linestyle='--')
 #   ax.axvline(pd.to_datetime(end_date), color='k', linestyle='--')

    plt.title("Unit scanned sales (000):"+query_name,fontsize=10)   #str(new_plot_df.columns.get_level_values(0)))
    plt.legend(fontsize=8)
    plt.ylabel("(000) units scanned/week")
    plt.grid(True)
 #   self.save_fig("actual_"+query_name+query_details,self.images_path)
    plt.show()

     
        
        
#column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
 
col_dict=dict({0:"Coles_scan_week",
               1:"BB_off_promo_sales",
               2:"BB_on_promo_sales",
               3:"SD_off_promo_sales",
               4:"SD_on_promo_sales",
               5:"BM_off_promo_sales",
               6:"BM_on_promo_sales"})
       
df=pd.read_excel(colesjamsxls,-1,skiprows=[0,1,2]).T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
#print("before",df)
df = df.rename(col_dict,axis='index').T

df['Coles_scan_week']=pd.to_datetime(df['Coles_scan_week'],format="%d/%m/%y")
#df['coles_scan_week']=df["date"] #.strftime("%Y-%m-%d")
df.fillna(0.0,inplace=True)
df.drop_duplicates(keep='first', inplace=True)
#df.replace(0.0, np.nan, inplace=True)
#print("after",df)

df=df.sort_values(by=['Coles_scan_week'], ascending=True)
df=df.set_index('Coles_scan_week') 
df=df.astype(np.float32)  #,inplace=True)
df['weekno']= np.arange(len(df))
#print("final",df,df.T)

df['BB_on_promo']=(df['BB_on_promo_sales']>0.0)
df['SD_on_promo']=(df['SD_on_promo_sales']>0.0)
df['BM_on_promo']=(df['BM_on_promo_sales']>0.0)

df['BB_total_sales']=df['BB_off_promo_sales']+df['BB_on_promo_sales']
df['SD_total_sales']=df['SD_off_promo_sales']+df['SD_on_promo_sales']
df['BM_total_sales']=df['BM_off_promo_sales']+df['BM_on_promo_sales']

    


df.replace(0.0, np.nan, inplace=True)

#print("df=",df)
plot_query(df,['BB_total_sales','SD_total_sales','BM_total_sales'],'BB total scanned Coles jam units per week')



#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')  #,fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BB_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='BB_on_promo')

sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')

