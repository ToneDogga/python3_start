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


# logfile="IRI_reader_logfile.txt"
# resultsfile="IRI_reader_results.txt"
# pklsave="IRI_savenames.pkl"
# colnamespklsave="IRI_savecoldetails.pkl"
# fullcolnamespklsave="IRI_saveallcoldetails.pkl"
# dfdictpklsave="IRI_savedfdict.pkl"
# dfpklsave="IRI_fullspreadsheetsave.pkl"

wwjamsxls="Coles_jams_scan_data_300520.xlsx"




#def plot_df(df,title,col_focus_no):    

     # print("Corr matrix1",title,"\n",df1.corr(),"\n\n")  
     # scatter_matrix(df1,alpha=1,figsize=(12,9))
     # print("Corr matrix2",title,"\n",df1.corr(),"\n\n")  
     # scatter_matrix(df2,alpha=1,figsize=(12,9))
     # print("Corr matrix3",title,"\n",df1.corr(),"\n\n")  
     # scatter_matrix(df3,alpha=1,figsize=(12,9))

 #    ax=df[df.columns[:3]].plot(x=df.columns[0],y=df.columns[1],style=["b:","r:","k:"],kind='scatter')   # actual
     #plot_numbdf[plot_number_df.columns[col_no]].plot(yerr=plot_number_df[plot_number_df.columns[col_no+1]],style='r', ecolor=['red'],errorevery=10)
  #   ax.axvline(pd.to_datetime(start_date), color='k', linestyle='--')
  #   ax.axvline(pd.to_datetime(end_date), color='k', linestyle='--')
    
  #    plt.title(title)   #str(new_plot_df.columns.get_level_values(0)))
  #    plt.legend(fontsize=11)
  #    plt.ylabel("units/week sales")
  #    plt.grid(True)
  # #   self.save_fig("actual_v_prediction_"+query_names[query_number],self.images_path)
    
  #    plt.show()
   #   if col_focus_no==0: 
   #       print("BB promos")
   #       df1=df.iloc[:,:3]
   # #  df2=df.iloc[:,1:3]
   #       df2=df.iloc[:,0::2]
   #   elif col_focus_no==1: 
   #       print("sd promos")
  #  df1=df.iloc[:,:5]
#    df2=df.iloc[:,1:3]
#    df3=df.iloc[:,0::2]
     # elif col_focus_no==2: 
     #     print("bm promos")
     #  #   df1=df.iloc[:,:2]
     #     df1=df.iloc[:,1:4]
     #     df2=df.iloc[:,0::2]
   
   #  print("df",df)
  #   print("df1",df1)
  #   print("df2",df2)
  #   print("df3",df3)
 #    df1.plot.scatter(x=df1.columns[0],y=df1.columns[1],title=title)
 #    df2.plot.scatter(x=df2.columns[0],y=df2.columns[1],title=title)
 #    df3.plot.scatter(x=df3.columns[0],y=df3.columns[1],title=title)


  #  sns.lmplot(x=df.columns[0],y=df.columns[1],col=df.columns[3],data=df,hue=df.columns[3],fit_reg=True,robust=True,legend=True) 
 #    sns.lmplot(x=df2.columns[0],y=df2.columns[1],data=df2,hue=df1.columns[2],fit_reg=True,robust=True,legend=True) 
  #   sns.lmplot(x=df3.columns[0],y=df3.columns[1],data=df3,fit_reg=True,robust=True) 



# def save_df(df,filename):
#     df.to_pickle(filename)
#     return


# def load_df(filename):
#     return pd.read_pickle(filename)



# class salestrans:
#     def __init__(self):   
#         self.epochs=8
#     #    self.steps_per_epoch=100 
#         self.no_of_batches=1000
#         self.no_of_repeats=2
        
#         self.dropout_rate=0.2
#         self.start_point=0
        
 
def add_dates_back(df,all_dates):
    return pd.concat((df,all_dates_df),axis=1)   #,on='ww_scan_week',how='right')

       
        
        
#column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
 
col_dict=dict({0:"coles_scan_week",
               1:"BB_off_promo_sales",
               2:"BB_on_promo_sales",
               3:"SD_off_promo_sales",
               4:"SD_on_promo_sales",
               5:"BM_off_promo_sales",
               6:"BM_on_promo_sales"})
       
df=pd.read_excel(wwjamsxls,-1,skiprows=[0,1,2]).T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
#print("before",df)
df = df.rename(col_dict,axis='index').T

df['coles_scan_week']=pd.to_datetime(df['coles_scan_week'],format="%d/%m/%y")
#df['coles_scan_week']=df["date"] #.strftime("%Y-%m-%d")
df.fillna(0.0,inplace=True)
df.drop_duplicates(keep='first', inplace=True)
#df.replace(0.0, np.nan, inplace=True)
#print("after",df)

df=df.sort_values(by=['coles_scan_week'], ascending=True)
df=df.set_index('coles_scan_week') 
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

