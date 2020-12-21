#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:06:18 2020

@author: tonedogga
"""


# =============================================================================
# 
# #  architecture                                inputs                         outputs
#------------------------------------------------------------------------------------------------
# #  dash2_class.sales.load                   list of filenames              a pandas df
# #  dash2_class.sales.preprocess                   df                            df
# #  dash2_class.sales.save                         df                       a df pickled file name
# 
# #  dash2_class.sales.query.load              a df pickled filename              df  
# #  dash2_class.sales.query.preprocess             df                           df
# #  dash2_class.sales.query.query          df, query dict                a directory of pickled df's           
# 
# #  dash2_class.sales.pivot.load          query dict, a dir of pkl df's         a dict of df's
# #  dash2_class.sales.pivot.preprocess           a dict of dfs                  a dict of df's
# #  dash2_class.sales.pivot.pivot           a dict of dfs, pivot_desc      a dict of df's   
# #  dash2_class.sales.pivot.save            a dict of df's                 a directory of excel spreadsheets
# 
# #  dash2_class.sales.plot.load           query dict, a dir of pkl dfs          a dict of df's
# #  dash2_class.sales.plot.preprocess            a dict of df's                 a dict of df's
# #  dash2_class.sales.plot.mat                  a dict of df's                a output dir of plots  
# #  dash2_class.sales.plot.compare_customer
# #  dash2_class.sales.plot.trend
# #  dash2_class.sales.plot.yoy_customer
# #  dash2_class.sales.plot.yoy_product
# #  dash2_class.sales.plot.pareto_customer
# #  dash2_class.sales.plot.pareto_product
# 
# #  dash2_class.sales.predict.load           query dict, a dir of pkl dfs          a dict of df's
# #  dash2_class.sales.predict.preprocess            a dict of df's                 a dict of df's
# #  dash2_class.sales.predict.next_week                X_train,y_train, validate, test     df
# #  dash2_class.sales.predict.actual_vs_expected
# 
# 
# =============================================================================


import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

import datetime as dt
from datetime import datetime
from datetime import date
from datetime import timedelta
import calendar
import xlsxwriter
import matplotlib.pyplot as plt
import xlrd

import joblib
from natsort import natsorted

from collections import OrderedDict
from collections import namedtuple
from collections import defaultdict

from matplotlib import pyplot, dates
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show

import matplotlib.cm as cm
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

import sklearn.linear_model
import sklearn.neighbors

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from pandas.plotting import scatter_matrix


import pickle
import subprocess as sp

import os
os.chdir("/home/tonedogga/Documents/python_dev")
cwdpath = os.getcwd()

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

tf.autograph.set_verbosity(0, False)
import subprocess as sp

from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"



import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

from pathlib import Path

from p_tqdm import p_map,p_umap

import dash2_dict as dd2
#import sales_trans_lib_v6



class sales_class(object):
    def __init__(self,df):
      #  query=sales_query_class("tesr")
       # self.sales_init="sales_init called"
       # print(self.sales_init)
    #    
       # df=pd.DataFrame([])
        self.query=sales_query_class(df)
        
        self.preprocess=self.query.preprocess(pd.DataFrame([]),{})   #"tesdtd")
        self.queries=self.query.queries(df)    
        #   self.load_sq=sales_query_class()   #.preprocess("tesdtd","dd")
    #    self.save=sales_query_class()   #.preprocess("tesdtd","dd")

        self.pivot=sales_pivot_class()   #.preprocess("tesdtd","dd")
        self.plot=sales_plot_class()   #.preprocess("tesdtd","dd")
        self.predict=sales_predict_class()   #.preprocess("tesdtd","dd")
 
             
  
   
    def _load_excel(self,filename):
      #  filename=dd2.dash2_dict['sales']['in_dir']+filename
        print("load:",filename)
        new_df=pd.read_excel(filename,sheet_name="AttacheBI_sales_trans",usecols=range(0,17),verbose=False)  # -1 means all rows  
        new_df['date'] = pd.to_datetime(new_df.date)  #, format='%d%m%Y')
        return new_df

    
    
    def load_from_excel(self,in_dir,filenames_list):  # filenames is a list of xlsx files to load and sort by date
       # os.makedirs(in_dir, exist_ok=True)
        filenames_list=[in_dir+f for f in filenames_list]
      #  print(filenames_list)
        
        df=pd.DataFrame([])
        df=df.append(p_umap(self._load_excel,filenames_list)) 
        
        df.fillna(0,inplace=True)
        df=df[(df.date.isnull()==False)]
        df.drop_duplicates(keep='first', inplace=True)
        df.sort_values(by=['date'], inplace=True, ascending=False)
   
        df["period"]=df.date.dt.to_period('D')
    
        df['period'] = df['period'].astype('category')
        df.set_index('date',inplace=True,drop=False) 
        return df   #.rename(columns=qd.rename_columns_dict,inplace=True)

          
     
    def preprocess_sc(self,df, rename_dict):
     #   print("preprocess_sc sales save df=",df,rename_dict)
        df=df[(df['code']!="OFFINV")]   
        df=df[(df['product']!="OFFINV")]   
        
        df.to_pickle(dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['raw_savefile'],protocol=-1)
 
    #---------------------------------------------------------
 
       # if dd2.dash2_dict['sales']['glset_not_spc_mask_flag']:
       #     new_df=df[((df['productgroup'].isin(dd2.dash2_dict['sales']['pg_only'])) & (df['glset'].isin(dd2.dash2_dict['sales']['glset_only'])))].copy()  
       # else:    
        if dd2.dash2_dict['sales']['apply_mask']:  # apply masking                                                                                          
            df=df[((df['productgroup'].isin(dd2.dash2_dict['sales']['pg_only'])) & (df['specialpricecat'].isin(dd2.dash2_dict['sales']['spc_only'])))]  

        new_df=df[~df['product'].isin(dd2.dash2_dict['sales']["GSV_prod_codes_to_exclude"])].copy()    
 
    
#------------------------------------------------------------------------


        print("\nPreprocess data to convert GSV to NSV exclude products=",dd2.dash2_dict['sales']["GSV_prod_codes_to_exclude"])
       # df=df[(df['code']=="OFFINV") | (df['product']=="OFFINV")]
        new_df.rename(columns=rename_dict, inplace=True)
     #   df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        #print("rename collumsn"
    #    print("preprocesed=\n",new_df.head(100))
        return new_df
    
      
    def _preprocess(self,df, rename_dict):
     #   print("preprocess_sc sales save df=",df,rename_dict)
        df=df[(df['code']!="OFFINV")]   
        df=df[(df['product']!="OFFINV")]   
        
     #   df.to_pickle(dd2.dash2_dict['sales']['raw_savefile'],protocol=-1)
 
    #---------------------------------------------------------
 
       # if dd2.dash2_dict['sales']['glset_not_spc_mask_flag']:
       #     new_df=df[((df['productgroup'].isin(dd2.dash2_dict['sales']['pg_only'])) & (df['glset'].isin(dd2.dash2_dict['sales']['glset_only'])))].copy()  
       # else:  
        if dd2.dash2_dict['sales']['apply_mask']:  # nasking to apply                                                                                                    
            df=df[((df['pg'].isin(dd2.dash2_dict['sales']['pg_only'])) & (df['spc'].isin(dd2.dash2_dict['sales']['spc_only'])))]  

        new_df=df[~df['product'].isin(dd2.dash2_dict['sales']["GSV_prod_codes_to_exclude"])].copy()    
 
    
#------------------------------------------------------------------------


        print("\nPreprocess data to convert GSV to NSV exclude products=",dd2.dash2_dict['sales']["GSV_prod_codes_to_exclude"])
       # df=df[(df['code']=="OFFINV") | (df['product']=="OFFINV")]
        new_df.rename(columns=rename_dict, inplace=True)
     #   df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        #print("rename collumsn"
    #    print("preprocesed=\n",new_df.head(100))
        return new_df
    
    
      
        
    def save(self,sales_df,save_dir,savefile):
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(sales_df, pd.DataFrame):
            if not sales_df.empty:
               # sales_df=pd.DataFrame([])
               sales_df.to_pickle(save_dir+savefile,protocol=-1)
               return True
            else:
               return False
        else:
            return False
     
    
        


    def load(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            return pd.read_pickle(save_dir+savefile)
        else:
            print("load sales_df error.")
            return
        
    
        
    
    def report(self,df):
        print("\nsales_df Report: Preprocess masking excluded OFFINV prod and cust codes. \nInclude only product groups=",dd2.dash2_dict['sales']["pg_only"],"\n and special price cat=",dd2.dash2_dict['sales']['spc_only'])
      #  df=dash.sales.query.load(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['savefile'])
        latest_date=pd.to_datetime(df['date'].iloc[0])
        first_date=pd.to_datetime(df['date'].iloc[-1])  #df['date'].iloc[-1]
        
        
        print("\nAttache sales trans analysis up to date.  New save is:",dd2.dash2_dict['sales']['savefile'])
        print("Data available:",df.shape,"records.\nfirst date:",first_date.strftime('%d/%m/%Y'),"\nlatest date:",latest_date.strftime('%d/%m/%Y'),"\n")
        return first_date,latest_date
    
       
    
    
    
    
    
        
    
    def _glset_NSV(self,dds,title):
    #    dds.index =  pd.to_datetime(dds['date'], format='%Y%m%d') 
    
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        dds['mat7']=dds['salesval'].rolling(7,axis=0).sum()
        dds['mat14']=dds['salesval'].rolling(14,axis=0).sum()
        dds['mat30']=dds['salesval'].rolling(30,axis=0).sum()
        dds['mat90']=dds['salesval'].rolling(90,axis=0).sum()
    
      #  dds['diff1']=dds.mat.diff(periods=1)
        dds['diff7']=dds.mat7.diff(periods=7)
        dds['diff14']=dds.mat14.diff(periods=14)
        dds['diff30']=dds.mat30.diff(periods=30)
        dds['diff90']=dds.mat90.diff(periods=90)
        dds['diff365']=dds.mat.diff(periods=365)
        
        dds['7_day%']=round(dds['diff7']/dds['mat7']*100,2)
        dds['14_day%']=round(dds['diff14']/dds['mat14']*100,2)
        dds['30_day%']=round(dds['diff30']/dds['mat30']*100,2)
        dds['90_day%']=round(dds['diff90']/dds['mat90']*100,2)
        dds['365_day%']=round(dds['diff365']/dds['mat']*100,2)
        
        dds['date']=dds.index.tolist()
  #      latest_date=dds['date'].max()
        dds.reset_index(inplace=True)
        #self.dash.sales.plot.mat({title:dds},dd2.dash2_dict['sales']['annual_mat'],latest_date,plot_output_dir)
        return dds[['dates','mat7','diff7','30_day%','90_day%','365_day%','mat']].tail(8)
        
        
        
    
       
    
    
    def summary(self,in_dir,raw_savefile):
        all_raw_dict={}
        
        sales_df=pd.read_pickle(in_dir+raw_savefile)
        sales_df.rename(columns={"specialpricecat":"spc","productgroup":"pg"}, inplace=True)
        name="DFS NSV sales $"
        print("\n",name)
        shop_df=sales_df[(sales_df['glset']=="DFS")]
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_dfs']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))
  
        name="shop NSV sales $"
        print("\n",name)
        shop_df=sales_df[(sales_df['glset']=="SHP")].copy()
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_shop']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))
        
    
        name="Online NSV sales $"
        print("\n",name)
        shop_df=sales_df[(sales_df['glset']=='ONL')].copy()
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_onl']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))

           
        name="Export NSV sales $"
        print("\n",name)
        shop_df=sales_df[(sales_df['glset']=="EXS")]
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_exs']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))


        name="NAT sales NSV$"
        print("\n",name)
        shop_df=sales_df[(sales_df['glset']=="NAT")]
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_nat']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))
        
        
        name="WW (010) NSV sales $"
        print("\n",name)
        shop_df=sales_df[(sales_df['spc']==10)].copy()
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
 #       shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_ww']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))


        name="Coles (012) NSV sales $"
        print("\n",name)
        shop_df=sales_df[(sales_df['spc']==12)].copy()
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_coles']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))
        
 
        name="Beerenberg NSV Annual growth rate"
        print("\n",name)
        shop_df=sales_df.copy()
        dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
        dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
        shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
        all_raw_dict['raw_all']=shop_df.copy()
        dds['dates']=dds.index.tolist()
        print(self._glset_NSV(dds,name))
        print("\n\n")
        print("============================================================================\n")  
 
        del sales_df
         #  sales_df.rename(columns=rename_dict, inplace=True)
        return all_raw_dict
 
    
 
    
    
    def pivot(self,df,pivot_desc):
        self.pivot=sales_pivot_class()
        print("pivot table",df,"pivot desc=",pivot_desc)
        return # excel file name if rows>0
        
    
    def plot(self,df,plottype):
        print("plot in sales class=")
        self.plot=sales_plot_class("")
        # plot type 
        return
    
    
    def predict(self,df,predicttype):
        print("predict in sales class=",df)
        self.predict=sales_predict_class()
        # predict type
        return
    
    
 
    
    
    
class sales_query_class(object):
    
    
    def __init__(self,df):   #,in2file):
    
     
     #   df=pd.DataFrame([])
       # self.sales_query_init="sales_query_init called"
       # print(self.sales_query_init)
     #   print("infile",in2file) 
      #  print("sales query class init")
     #   self.load_sq=self.load_sq("df") 
      #  self.preprocess=self.preprocess("df") 
        pass
        
        
        
    # def load(self,save_dir,infile):
    #     df=pd.read_pickle(save_dir+infile)
    #    # print("load sales queryclass=",df)
    #     return df


    def preprocess(self,df,rename_dict):
       # print("preprocess sales query df=",df)
     #   print("\nPreprocess data exclude OFFINV. \nInclude only product groups=",dd2.dash2_dict['sales']["pg_only"],"\nSpecial price cat=",dd2.dash2_dict['sales']['spc_only'])

        pass 
     #   df.rename(columns=rename_dict, inplace=True)
      #  df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
    #    print("rename collumsn")
        return df
      
 
        
        
        
        
    def _query_df(self,new_df,query_name):
# =============================================================================
#         
#         #   query of AND's - input a list of tuples.  ["AND",(field_name1,value1) and (field_name2,value2) and ...]
#             the first element is the type of query  -"&"-AND, "|"-OR, "!"-NOT, "B"-between
# #            return a slice of the df as a copy
# # 
# #        a query of OR 's  -  input a list of tuples.  ["OR",(field_name1,value1) or (field_name2,value2) or ...]
# #            return a slice of the df as a copy
# #
# #        a query_between is only a triple tuple  ["BD",(fieldname,startvalue,endvalue)]
#                "BD" for between dates, "B" for between numbers or strings
# # 
# #        a query_not is only a single triple tuple ["NOT",(fieldname,value)]   
# 
#         
# =========================================================================
 #    print("query_df df=\n",new_df,"query_name=",query_name)  
     if (query_name==[]) | (new_df.shape[0]==0):
           return new_df
     else :   
           if ((query_name[0]=="AND") | (query_name[0]=='OR') | (query_name[0]=="BD")| (query_name[0]=="B") | (query_name[0]=="NOT")):
                oper=str(query_name[0])
             #   print("valid operator",oper,new_df.shape)
                query_list=query_name[1:]
  
                
       #         new_df=df.copy()
                if oper=="AND":
                 #   print("AND quwery_list",query_list)
                    for q in query_list:  
                        field=str(q[0])
                        new_df=new_df[(new_df[field]==q[1])].copy() 
                  #      print("AND query=",field,"==",q[1],"\nnew_df=",new_df.shape) 
                  #      print("new new_df=\n",new_df)    
                elif oper=="OR":
                    new_df_list=[]
                    for q in query_list:    
                        new_df_list.append(new_df[(new_df[q[0]]==q[1])].copy()) 
                     #   print("OR query=",q,"|",new_df_list[-1].shape)
                    new_df=new_df_list[0]    
                    for i in range(1,len(query_list)):    
                        new_df=pd.concat((new_df,new_df_list[i]),axis=0)   
                  #  print("before drop",new_df.shape)    
                    new_df.drop_duplicates(keep="first",inplace=True)   
                  #  print("after drop",new_df.shape)
                elif oper=="NOT":
                    for q in query_list:    
                        new_df=new_df[(new_df[q[0]]!=q[1])].copy() 
                   #     print("NOT query=",q,"NOT",new_df.shape)  
                   
                  #   new_df_list=[]
                  #   for q in query_list:    
                  #       new_df_list.append(new_df[(new_df[q[0]]!=q[1])].copy()) 
                  #    #   print("OR query=",q,"|",new_df_list[-1].shape)
                  #   new_df=new_df_list[0]    
                  #   for i in range(1,len(query_list)):    
                  #       new_df=pd.concat((new_df,new_df_list[i]),axis=0)   
                  # #  print("before drop",new_df.shape)    
                  #   new_df.drop_duplicates(keep="first",inplace=True)   
    
                   
                elif oper=="BD":  # betwwen dates
                  #  if (len(query_list[0])==3):
                    for q in query_list:
                    #    print("between ql=",q[1],q[2])
                        start=q[1]
                        end=q[2]
                        new_df=new_df[(pd.to_datetime(new_df[q[0]])>=pd.to_datetime(q[1])) & (pd.to_datetime(new_df[q[0]])<=pd.to_datetime(q[2]))].copy() 
                     #       print("Beeterm AND query=",q,"&",new_df.shape) 
                   # else:
                   #     print("Error in between statement")
                elif oper=="B":  # btween numbers or strings
                  #  if (len(query_list[0])==3):
                    for q in query_list:
                    #    print("between ql=",q[1],q[2])
                        start=q[1]
                        end=q[2]
                        new_df=new_df[(new_df[q[0]]>=q[1]) & (new_df[q[0]]<=q[2])].copy() 
                     #       print("Beeterm AND query=",q,"&",new_df.shape) 
                   # else:
                   #     print("Error in between statement")
     
                else:
                    print("operator not found\n")
                        
                return new_df.copy()
                      
           else:
                print("invalid operator")
                return pd.DataFrame([])
    
  
      
    def _build_a_query_dict(self,query_name):
     #   print("build an entry query_name",query_name)
     #   print("query name=",query_name[1])
     #   print("filesave name=",query_name[0])
        #queries=query_name[1]
      #  query_name=qd.queries[q]
        new_df=dd2.dash2_dict['sales']['query_df']
      #  new_df=query_df.copy()
        for qn in query_name[1]:  
        #    print("build a query dict qn=",qn)
            q_df=self._query_df(new_df,qn)
            new_df=q_df.sort_index(ascending=False,axis=0).copy()
        q_df.drop_duplicates(keep="first",inplace=True)    
       # q_df=smooth(q_df)
        self.save(q_df,dd2.dash2_dict['sales']['save_dir'],query_name[0])   
        return q_df
    
           
    
    def queries(self,qdf):
      #  self.query=sales_query_class()
      
        query_df=qdf.copy()
        dd2.dash2_dict['sales']['query_df']=query_df.copy()
        if query_df.shape[0]>0:
         #   df=df.rename(columns=qd.rename_columns_dict)  
          #  query_handles=[]
            query_filenames=[]
            query_filenames.append(p_map(self._build_a_query_dict,dd2.dash2_dict['sales']['queries'].items()))   #st.save_query(q_df,query_name,root=False)   
         #   query_filenames=[q[:250] for q in query_handles[0]]  # if len(q)>249]
         #   print("build a query dict query filenames",query_filenames)
            return {k: v for k, v in zip(dd2.dash2_dict['sales']['queries'].keys(),query_filenames[0])}     #,{k: v for k, v in zip(qd.queries.keys(),query_filenames)}
        else:
            print("df empty",query_df)
            return {}
    
    

    def save(self,df_dict,save_dir,savefile):
        os.makedirs(save_dir, exist_ok=True)
      #  df_dict.to_pickle(save_dir+savefile,protocol=-1)
        with open(save_dir+savefile+".pkl",'wb') as handle:
            pickle.dump(df_dict, handle, protocol=-1)
        
   
    # def query_dict_load(self,save_dir,keys):
    #     df_dict={}
    #     for k in keys:
    #         with open(save_dir+k+".pkl",'rb') as handle:
    #             df=pickle.load(handle)
    #         df_dict[k]=df    
    #     return df_dict

        
     

        
          
    
class sales_pivot_class(object):
    def __init__(self):  #,in3file):
      #  self.sales_pivot_init="sales_pivot_init called"
      #  print(self.sales_pivot_init)
 #       print("sales pivot class init")
       # print("infile",in3file)   
        pass
        

    def load(self,infile):
        print("load sales pivot query=",infile)
        return("returned sales pivot query load")
        
        
        
  #  def pivot_df(self,infile,desc):
  #      print("pivot df pivot class=",infile,desc)
  #      return("returned sales query load")
         
     
    # def preprocess(self,df, rename_dict):
    #     # rename qolumns
    #     print("preprocess pivot class=",df,rename_dict)

    #     return df
      
        
    def save(self,df):
        print("save pivot class")
        return("sales save outfile")



         
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
    
    
    def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
    


    def negative(self,df):
        df=df.copy()
        df['salesval']*=-1
        df['qty']*=-1
        return df
    
    
    def _multi_calc_trend(self,distribution_details):
        # s=s.sort_values('date',ascending=True)
         dist_df=dd2.dash2_dict['sales']['dist_sales_df']
      #   latest_date=distribution_details['date'].max()
     #    dist_df=dd2.dash2_dict['sales']['dist_df'].copy()
         
         cust=distribution_details[0]
         prod=distribution_details[1]
         dist_df=distribution_details[2]
   #      qty_sum=distribution_details[4]
   #      salesval_sum=distribution_details[5]
   #      most_recent_date=distribution_details[3]
         
    #     latest_date=pd.to_datetime(dist_df['date'].max()).strftime("%d/%m/%Y")
         if isinstance(dist_df,pd.DataFrame):
             if dist_df.shape[0]>0:
              #   print("yes",dist_df.shape)
                 lastone=dist_df.iloc[-1]
              #   print("1distdf=\n",dist_df)
        
                 newgap=pd.Timedelta(pd.to_datetime('today') -lastone['date']).days
         
                 today_date=pd.to_datetime('today') 
                 dist_df=dist_df.append(lastone)
       
                 
                 dist_df['date'].iloc[-1]=pd.to_datetime('today')
               
                 dist_df['daysdiff']=dist_df['date'].diff(periods=1).dt.days
                 dist_df['daysdiff'].fillna(0,inplace=True)
            
                 dist_df['days']=dist_df['daysdiff'].cumsum()
            
               #  print("3distdf=\n",dist_df)
        
              #   dist_df.index =  pd.to_datetime(dist_df['date'], format='%Y%m%d') 
                 X=dist_df[['days']].to_numpy()
                 X=X[::-1,0]
              #  X=X[:,0]
            
                 dist_df['days since last order']=X
                 y=dist_df[['qty']].to_numpy()   
                 y=y[::-1,0]
                 dist_df['units']=y
               
                 p = np.polyfit(X[:-1], y[:-1], 1)  # linear regression 1 degree
                
                 dist_df['bestfit']=np.polyval(p, X)
              #   print("4distdf=\n",dist_df)
        
               # print("s=",s)
              #   figname=""
              #   title=""
                 slope=round(p[0],6)
               # print("slope=",slope)
                 if ((slope>dd2.dash2_dict['sales']['plots']['max_slope']) | (slope<dd2.dash2_dict['sales']['plots']['min_slope'])):
                #    print("\nPPPPP plot slope=",slope)
                  
                   #  title="trend_"+str(round(slope,3))+"_"+str(cust)+"_"+str(prod)
                   #  sns.lmplot(x='days since last order',y='units', hue='on_promo',data=dist_df)    # col='on_promo_guess',
                 
                  #   plt.title(title+" (slope="+str(round(slope,3))+") w/c:"+str(latest_date))  #str(new_plot_df.columns.get_level_values(0)))
                  #   fig.legend(fontsize=8)
                  #   plt.ylabel("unit sales")
                  #   plt.grid(True)
                    #     self.save_fig("actual_v_prediction_"+str(plot_number_df.columns[0]),self.images_path)
                  #   figname=title
                     
                  #   self._save_fig(self._clean_up_name(figname),dd2.dash2_dict['sales']['plot_output_dir'])
                     return [cust,prod,dist_df,slope]
                 else:
                     return [cust,prod,pd.DataFrame([]),0.0]
             else:
                  return [cust,prod,pd.DataFrame([]),0.0]
         else:
             return [cust,prod,pd.DataFrame([]),0.0]
   
    
    
     
    def _multi_plot_trend(self,distribution_details):
        # s=s.sort_values('date',ascending=True)
        
      #   latest_date=distribution_details['date'].max()
      #   dist_df=dd2.dash2_dict['sales']['dist_df'].copy()
         dist_df=dd2.dash2_dict['sales']['dist_sales_df']
         
         cust=distribution_details[0]
         prod=distribution_details[1]
         dist_df=distribution_details[2]
         slope=distribution_details[3]
        # salesval_sum=distribution_details[5]
        # most_recent_date=distribution_details[3]
         
         latest_date=pd.to_datetime(dist_df['date'].max())    #.strftime("%d/%m/%Y")
         if isinstance(dist_df,pd.DataFrame):
             if dist_df.shape[0]>0:
              #   print("yes",dist_df.shape)
                 lastone=dist_df.iloc[-1]
              #   print("1distdf=\n",dist_df)
        
                 newgap=pd.Timedelta(pd.to_datetime('today') -lastone['date']).days
         
                 today_date=pd.to_datetime('today') 
                 dist_df=dist_df.append(lastone)
       
                 
                 dist_df['date'].iloc[-1]=pd.to_datetime('today')
               
                 dist_df['daysdiff']=dist_df['date'].diff(periods=1).dt.days
                 dist_df['daysdiff'].fillna(0,inplace=True)
            
                 dist_df['days']=dist_df['daysdiff'].cumsum()
            
               #  print("3distdf=\n",dist_df)
        
              #   dist_df.index =  pd.to_datetime(dist_df['date'], format='%Y%m%d') 
                 X=dist_df[['days']].to_numpy()
                 X=X[::-1,0]
              #  X=X[:,0]
            
                 dist_df['days since last order']=X
                 y=dist_df[['qty']].to_numpy()   
                 y=y[::-1,0]
                 dist_df['units']=y
               
                 p = np.polyfit(X[:-1], y[:-1], 1)  # linear regression 1 degree
                
                 dist_df['bestfit']=np.polyval(p, X)
              #   print("4distdf=\n",dist_df)
        
               # print("s=",s)
                 figname=""
                 title=""
                 slope=round(p[0],6)
               # print("slope=",slope)
                 if ((slope>dd2.dash2_dict['sales']['plots']['max_slope']) | (slope<dd2.dash2_dict['sales']['plots']['min_slope'])):
                #    print("\nPPPPP plot slope=",slope)
                  
                     title="Trend_"+str(round(slope,3))+"_"+str(cust)+"_"+str(prod)
                     sns.lmplot(x='days since last order',y='units', hue='on_promo',data=dist_df)    # col='on_promo_guess',
                 
                     plt.title(title+" (slope="+str(round(slope,3))+") w/c:"+latest_date.strftime('%d/%m/%Y'))  #str(new_plot_df.columns.get_level_values(0)))
                 #    fig.legend(fontsize=8)
                     plt.ylabel("unit sales")
                     plt.grid(True)
                   #  self.save_fig("actual_v_prediction_"+str(plot_number_df.columns[0]),self.images_path)
                     figname=title
                     
                     self._save_fig(self._clean_up_name(figname),dd2.dash2_dict['sales']['plot_output_dir'])
                     return 
        
    
     





    
    
    
    def _multi_function_sales_slice(self,cust_and_prod):
        dist_sales_df=dd2.dash2_dict['sales']['dist_sales_df']
        new_df=dist_sales_df[(dist_sales_df['code']==cust_and_prod[0][2]) & (dist_sales_df['product']==cust_and_prod[1][1])]  #work_in_dict['split_key']]
        if new_df.shape[0]>=dd2.dash2_dict['sales']['plots']['min_size_for_trend_plot']:
            return [cust_and_prod[0],cust_and_prod[1],new_df,new_df['date'].max(),new_df['qty'].sum(),new_df['salesval'].sum(),0,0]    #sales_df[(sales_df['code']==cust_and_prod[0]) & (sales_df['product']==cust_and_prod[1])]]  #work_in_dict['split_key']]
        else:
            return []
    
      
    
   #  def _heatmap_to_excel_dollars(self,pivot_dist_df,output_dir):
   #  #    pivot_salesval_change_df.sort_values([("","All")],ascending=False,axis="index",inplace=True)
 
       
   #      sheet_name = 'Sheet1'

   #      writer = pd.ExcelWriter(output_dir+dd2.dash2_dict['sales']['pivots']["distribution_report_dollars"],engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')


   #      pivot_dist_df.to_excel(writer, sheet_name=sheet_name)
  
    
   #      workbook = writer.book
   #      worksheet = writer.sheets[sheet_name]
   #      money_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': False})
   #      total_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': True})

   #      worksheet.set_column('E:ZZ', 12, money_fmt)
   #      worksheet.set_column('D:D', 12, total_fmt)
   #      worksheet.set_row(3, 12, total_fmt)

   #          # Apply a conditional format to the cell range.
   # #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
   #      worksheet.conditional_format('E5:ZZ1000', {'type': '3_color_scale'})

   #      # Close the Pandas Excel writer and output the Excel file.
   #      writer.save()      
   #      return



    def _trend_heatmap(self,trends,output_dir):
      #   print("\ntrend heatmap\n",len(trends))    
         trends_copy=trends.copy()
      #   for t in trends_copy:
      #       del t[2]
         if len(trends_copy)>0:
             trend_df=pd.DataFrame(trends_copy).copy()    
          #   print("trend_df=\n",trend_df)
             if isinstance(trend_df,pd.DataFrame):
                 if trend_df.shape[0]>0:
                     pv=pd.pivot_table(trend_df,values=2,index=[0],columns=[1],aggfunc=np.mean,dropna=True,margins=False)
                     
                     pv.index = pd.MultiIndex.from_tuples(pv.index,names=('salesrep','spc','code'))
                     
                     pv=pv.T
                     pv.index = pd.MultiIndex.from_tuples(pv.index,names=('pg','product'))
                     pv=pv.T
                     
                     pv=pv.rename(dd2.salesrep_dict,level='salesrep',axis='index')
                     pv=pv.rename(dd2.spc_dict,level='spc',axis='index')
                     pv=pv.rename(dd2.spcs_dict,level='spc',axis='index')
            
               
                     pv=pv.rename(dd2.productgroup_dict,level='pg',axis='columns')
                     pv=pv.rename(dd2.productgroups_dict,level='pg',axis='columns')   #.copy()
                  
                     sheet_name = 'Sheet1'
            
                     writer = pd.ExcelWriter(output_dir+dd2.dash2_dict['sales']['pivots']["trend_heatmap_filename"],engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
            
                     pv.to_excel(writer, sheet_name=sheet_name)
                
                     workbook = writer.book
                     worksheet = writer.sheets[sheet_name]
                     
                     value_fmt = workbook.add_format({'num_format': '#,##0.00', 'bold': False})
                    # total_fmt = workbook.add_format({'num_format': '#,##0.00', 'bold': True})
            
                     worksheet.set_column('D:ZZ', 12, value_fmt)
                   #  worksheet.set_column('D:D', 12, total_fmt)
                   #  worksheet.set_row(3, 12, total_fmt)
            
                        # Apply a conditional format to the cell range.
               #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
                     worksheet.conditional_format('D4:ZZ1000', {'type': '3_color_scale'})
            
                     writer.save()      
         return
        
        
        


    
    def _prods_and_custs(self,df):
         prod_list=list(set([tuple(r) for r in df[['pg', 'product']].to_numpy()]))
         cust_list=list(set([tuple(r) for r in df[['salesrep','spc', 'code']].to_numpy()]))
         return [[c,p] for c in cust_list for p in prod_list]
    


    def distribution_report_dollars(self,name,passed_df,output_dir,trend):
        print(name,"distribution report dollars",passed_df.shape)
        #for q in passed_dict.keys():
        #    print("key=",q)
      
        dist_sales_df=passed_df.copy()
        dd2.dash2_dict['sales']['dist_sales_df']=dist_sales_df
        cust_prod_list=self._prods_and_custs(dist_sales_df)
         
        print("Slicing",name,"sales_df from",len(cust_prod_list),"possible combinations of customer and product with multiprocessing.")
   
        multiple_results=[]
        multiple_results.append(p_map(self._multi_function_sales_slice,cust_prod_list))  # stops, journey and poolsize, epoch length and name of q
 
        distribution_list=[elem for elem in multiple_results[0] if len(elem)!=0]
        
        t_list=[]
        if trend:
            print("Calc",len(distribution_list),"trends...")
            t_list.append(p_map(self._multi_calc_trend,distribution_list)) 
    

      #  print("\nPlot trends finished.\n",trends)
            trends=[elem for elem in t_list[0] if elem[3]!=0.0]
       # print("\n2 not sorted Plot trends finished.\n",trends2)

            trends.sort(key=lambda x:x[3])
            
 #           self._trend_heatmap(trends,output_dir)
            
            new_trends=trends[:16]+trends[-16:][::-1]
        
            print("Plot",len(new_trends),"trends...")
            p_map(self._multi_plot_trend,new_trends)  
 
            for sublist in trends:
               del sublist[2]
  
    
            print("\nBest performers",trends[-40:-4][::-1])
            print("\nWorst performers",trends[4:40],"\n")
    
            
            self._trend_heatmap(trends[4:-4],output_dir)
     
      #  else:
        
        for sublist in distribution_list:
            del sublist[2]
 
     #  print("post del dist list=",distribution_list)
        if len(distribution_list)>0:
            dist_df=pd.DataFrame(distribution_list,columns=['cust','prod',"latestdate","qtysum","salesvalsum","a","b"])
            if isinstance(dist_df,pd.DataFrame):
                 if dist_df.shape[0]>0:
                    try: 
                        pivot_dist_df=pd.pivot_table(dist_df, values='salesvalsum', columns='prod',index='cust', aggfunc=np.sum, margins=True,dropna=False)
                    except:
                        pass
                    else:
                        pivot_dist_df=pivot_dist_df.rename({"All":(999,999,"All")},level=0,axis='index')
                        pivot_dist_df=pivot_dist_df.rename({"All":(999,"All")},level=0,axis='columns')
                       
                        pivot_dist_df.index=pd.MultiIndex.from_tuples(pivot_dist_df.index,sortorder=0,names=['salesrep','spc','code'])   #['productgroup','product'])
                       
                        pivot_dist_df=pivot_dist_df.T
                
                        pivot_dist_df.index=pd.MultiIndex.from_tuples(pivot_dist_df.index,sortorder=0,names=['pg','product'])
                    #   print("2pivot_salesval_df=\n",pivot_salesval_df)
                        pivot_dist_df=pivot_dist_df.T
                       
                        pivot_dist_df=pivot_dist_df.rename(dd2.salesrep_dict,level='salesrep',axis='index')
                        pivot_dist_df=pivot_dist_df.rename(dd2.spc_dict,level='spc',axis='index')
                   
                        pivot_dist_df=pivot_dist_df.rename(dd2.productgroup_dict,level='pg',axis='columns')
                        pivot_dist_df=pivot_dist_df.rename(dd2.productgroups_dict,level='pg',axis='columns')   #.copy()
                       
                        pivot_dist_df.sort_values([("","All")],ascending=False,axis="index",inplace=True)
                        pivot_dist_df.sort_values([(999,"","All")],ascending=False,axis="columns",inplace=True)
                
                #        self._heatmap_to_excel_dollars(pivot_dist_df,output_dir)
                        sheet_name = 'Sheet1'
                
                        writer = pd.ExcelWriter(output_dir+"distribution_"+name+"_"+dd2.dash2_dict['sales']['pivots']["distribution_report_dollars"],engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
                
                
                        pivot_dist_df.to_excel(writer, sheet_name=sheet_name)
                  
                    
                        workbook = writer.book
                        worksheet = writer.sheets[sheet_name]
                        money_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': False})
                        total_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': True})
                
                        worksheet.set_column('E:ZZ', 12, money_fmt)
                        worksheet.set_column('D:D', 12, total_fmt)
                        worksheet.set_row(3, 12, total_fmt)
                
                            # Apply a conditional format to the cell range.
                   #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
                        worksheet.conditional_format('E5:ZZ1000', {'type': '3_color_scale'})
                
                        # Close the Pandas Excel writer and output the Excel file.
                        writer.save()      

        return 
    
    
    
    def distribution_report_units(self,name,passed_df,output_dir):
        print(name,"distribution report units",passed_df.shape)
        #for q in passed_dict.keys():
        #    print("key=",q)
    
        dist_sales_df=passed_df.copy()
        dd2.dash2_dict['sales']['dist_sales_df']=dist_sales_df
        cust_prod_list=self._prods_and_custs(dist_sales_df)
         
        print("Slicing",name,"sales_df from",len(cust_prod_list),"possible combinations of customer and product with multiprocessing.")
   
        multiple_results=[]
        multiple_results.append(p_map(self._multi_function_sales_slice,cust_prod_list))  # stops, journey and poolsize, epoch length and name of q
 
        distribution_list=[elem for elem in multiple_results[0] if len(elem)!=0]
        
      #   t_list=[]
      #   if trend:
      #       print("Calc",len(distribution_list),"trends...")
      #       t_list.append(p_map(self._multi_calc_trend,distribution_list) ) 
    

      # #  print("\nPlot trends finished.\n",trends)
      #       trends=[elem for elem in t_list[0] if elem[3]!=0.0]
      #  # print("\n2 not sorted Plot trends finished.\n",trends2)

      #       trends.sort(key=lambda x:x[3])
         
      #       new_trends=trends[:16]+trends[-16:][::-1]
        
      #       print("Plot",len(new_trends),"trends...")
      #       p_map(self._multi_plot_trend,new_trends)  
 
      #       for sublist in trends:
      #          del sublist[2]
  
    
      #       print("\nBest performers",trends[-20:][::-1])
      #       print("Worst performers",trends[:20],"\n")
     
        
        for sublist in distribution_list:
            del sublist[2]
 
    
        if len(distribution_list)>0:
 
     #  print("post del dist list=",distribution_list)
            dist_df=pd.DataFrame(distribution_list,columns=['cust','prod',"latestdate","qtysum","salesvalsum","a","b"])
            if isinstance(dist_df,pd.DataFrame):
                 if dist_df.shape[0]>0:
                    try:     
                        pivot_dist_df=pd.pivot_table(dist_df, values='qtysum', columns='prod',index='cust', aggfunc=np.sum, margins=True,dropna=False)
                    except:
                        pass
                    else:
                        pivot_dist_df=pivot_dist_df.rename({"All":(999,999,"All")},level=0,axis='index')
                        pivot_dist_df=pivot_dist_df.rename({"All":(999,"All")},level=0,axis='columns')
                       
                        pivot_dist_df.index=pd.MultiIndex.from_tuples(pivot_dist_df.index,sortorder=0,names=['salesrep','spc','code'])   #['productgroup','product'])
                       
                        pivot_dist_df=pivot_dist_df.T
                
                        pivot_dist_df.index=pd.MultiIndex.from_tuples(pivot_dist_df.index,sortorder=0,names=['pg','product'])
                    #   print("2pivot_salesval_df=\n",pivot_salesval_df)
                        pivot_dist_df=pivot_dist_df.T
                       
                        pivot_dist_df=pivot_dist_df.rename(dd2.salesrep_dict,level='salesrep',axis='index')
                        pivot_dist_df=pivot_dist_df.rename(dd2.spc_dict,level='spc',axis='index')
                   
                        pivot_dist_df=pivot_dist_df.rename(dd2.productgroup_dict,level='pg',axis='columns')
                        pivot_dist_df=pivot_dist_df.rename(dd2.productgroups_dict,level='pg',axis='columns')   #.copy()
                       
                        pivot_dist_df.sort_values([("","All")],ascending=False,axis="index",inplace=True)
                        pivot_dist_df.sort_values([(999,"","All")],ascending=False,axis="columns",inplace=True)
                
                #        self._heatmap_to_excel_dollars(pivot_dist_df,output_dir)
                        sheet_name = 'Sheet1'
                
                        writer = pd.ExcelWriter(output_dir+"distribution_"+name+"_"+dd2.dash2_dict['sales']['pivots']["distribution_report_units"],engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
                
                
                        pivot_dist_df.to_excel(writer, sheet_name=sheet_name)
                  
                        workbook = writer.book
                        worksheet = writer.sheets[sheet_name]
                        value_fmt = workbook.add_format({'num_format': '#,##0', 'bold': False})
                        total_fmt = workbook.add_format({'num_format': '#,##0', 'bold': True})
                
                        worksheet.set_column('E:ZZ', 12, value_fmt)
                        worksheet.set_column('D:D', 12, total_fmt)
                        worksheet.set_row(3, 12, total_fmt)
                
                            # Apply a conditional format to the cell range.
                   #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
                        worksheet.conditional_format('E5:ZZ1000', {'type': '3_color_scale'})
                
                        # Close the Pandas Excel writer and output the Excel file.
                        writer.save()    
        return
  
  
      
    def distribution_report_dates(self,name,passed_df,output_dir):
        print(name,"distribution report dates",passed_df.shape)
        #for q in passed_dict.keys():
        #    print("key=",q)
       
        dist_sales_df=passed_df.copy()
        dd2.dash2_dict['sales']['dist_sales_df']=dist_sales_df
        cust_prod_list=self._prods_and_custs(dist_sales_df)
         
        print("Slicing",name,"sales_df from",len(cust_prod_list),"possible combinations of customer and product with multiprocessing.")
   
        multiple_results=[]
        multiple_results.append(p_map(self._multi_function_sales_slice,cust_prod_list))  # stops, journey and poolsize, epoch length and name of q
 
        distribution_list=[elem for elem in multiple_results[0] if len(elem)!=0]
        
      #   t_list=[]
      #   if trend:
      #       print("Calc",len(distribution_list),"trends...")
      #       t_list.append(p_map(self._multi_calc_trend,distribution_list) ) 
    

      # #  print("\nPlot trends finished.\n",trends)
      #       trends=[elem for elem in t_list[0] if elem[3]!=0.0]
      #  # print("\n2 not sorted Plot trends finished.\n",trends2)

      #       trends.sort(key=lambda x:x[3])
         
      #       new_trends=trends[:16]+trends[-16:][::-1]
        
      #       print("Plot",len(new_trends),"trends...")
      #       p_map(self._multi_plot_trend,new_trends)  
 
      #       for sublist in trends:
      #          del sublist[2]
  
    
      #       print("\nBest performers",trends[-20:][::-1])
      #       print("Worst performers",trends[:20],"\n")
     
        
        for sublist in distribution_list:
            del sublist[2]
 
    
 
        if len(distribution_list)>0:
            dist_df=pd.DataFrame(distribution_list,columns=['cust','prod',"latestdate","qtysum","salesvalsum","a","b"])
            if isinstance(dist_df,pd.DataFrame):
                 if dist_df.shape[0]>0:
                    try: 
                        pivot_date_df=pd.pivot_table(dist_df, values='latestdate', columns='prod',index='cust', aggfunc=np.max, margins=False,dropna=False)
                    except:
                        pass
                    else:
                        pivot_date_df.index=pd.MultiIndex.from_tuples(pivot_date_df.index,sortorder=0,names=['salesrep','spc','code'])   #['productgroup','product'])
                      #  print("1pivot_salesval_df=\n",pivot_salesval_df)
                        
                        pivot_date_df=pivot_date_df.T
                  
                        pivot_date_df.index=pd.MultiIndex.from_tuples(pivot_date_df.index,sortorder=0,names=['pg','product'])
                     #   print("2pivot_salesval_df=\n",pivot_salesval_df)
                        pivot_date_df=pivot_date_df.T
                        
                        pivot_date_df=pivot_date_df.rename(dd2.salesrep_dict,level='salesrep',axis='index')
                        pivot_date_df=pivot_date_df.rename(dd2.spc_dict,level='spc',axis='index')
                
                        pivot_date_df=pivot_date_df.rename(dd2.productgroup_dict,level='pg',axis='columns')
                        pivot_date_df=pivot_date_df.rename(dd2.productgroups_dict,level='pg',axis='columns').copy()
                
                     
                    
                        sheet_name = 'Sheet1'
                
                     #   writer = pd.ExcelWriter(output_dir+"distribution_report.xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
                        writer = pd.ExcelWriter(output_dir+"distribution_report_dates_"+name+"_"+dd2.dash2_dict['sales']['pivots']["distribution_report_dates"],engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
                
                    
                        pivot_date_df.to_excel(writer, sheet_name=sheet_name)
                        
                        workbook = writer.book
                        worksheet = writer.sheets[sheet_name]
                
                            # Apply a conditional format to the cell range.
                   #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
                        worksheet.conditional_format('D4:ZZ1000', {'type': '3_color_scale'})
                
                        # Close the Pandas Excel writer and output the Excel file.
                        writer.save()      
        return
    
 
    
 
    
 
    
 
    
 
    
 
    
 
   
        
        
        
        
        
        
        
        
        
        
        
   


    def report(self,all_sales_df,output_dir):
     #   print("\n=====================================================================\n")
        all_sales_df.sort_index(ascending=True,axis=0,inplace=True)
     #   print("all sales df=\n",all_sales_df)
        latest_date=all_sales_df['date'].max()
        print("latest salestrans date=",latest_date.strftime('%d/%m/%Y'))
        end_date=all_sales_df['date'].iloc[-1]- pd.Timedelta(2, unit='d')
        #print(end_date)
        #print("ysdf=",sales_df)
        year_sales_df=all_sales_df[all_sales_df['date']>end_date].copy()
        #print("ysdf=",year_sales_df)
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["code"]).transform(sum)
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
        #print("pv=",pivot_df)
        unique_code_pivot_df=pivot_df.drop_duplicates('code',keep='first')
        #unique_code_pivot_df=pd.unique(pivot_df['code'])
        name="Top 20 customers by $purchases in the last 2 days"
        print("\n",name)
        print(unique_code_pivot_df[['code','total_dollars']].head(20))
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
        
    #    unique_code_pivot_df[['code','total_dollars']].head(20).to_excel(output_dir+name+".xlsx") 
        
       
        
        
        
        
        
       # latest_date=sales_df['date'].max()
        
        end_date=all_sales_df['date'].iloc[-1]- pd.Timedelta(7, unit='d')
        #print(end_date)
        #print("ysdf=",sales_df)
        year_sales_df=all_sales_df[all_sales_df['date']>end_date].copy()
        #print("ysdf=",year_sales_df)
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["code"]).transform(sum)
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
        #print("pv=",pivot_df)
        unique_code_pivot_df=pivot_df.drop_duplicates('code',keep='first')
        #unique_code_pivot_df=pd.unique(pivot_df['code'])
        name="Top 20 customers by $purchases in the last 7 days"
        print("\n",name)
        print(unique_code_pivot_df[['code','total_dollars']].head(20))
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
        
     #   unique_code_pivot_df[['code','total_dollars']].head(30).to_excel(output_dir+name+".xlsx") 
        
        
        
        
        
       # latest_date=sales_df['date'].max()
        
        end_date=all_sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
        #print(end_date)
        #print("ysdf=",sales_df)
        year_sales_df=all_sales_df[all_sales_df['date']>end_date].copy()
        #print("ysdf=",year_sales_df)
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["code"]).transform(sum)
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
        #print("pv=",pivot_df)
        unique_code_pivot_df=pivot_df.drop_duplicates('code',keep='first')
        #unique_code_pivot_df=pd.unique(pivot_df['code'])
        name="Top 20 customers by $purchases in the last 30 days"
        print("\n",name)
        print(unique_code_pivot_df[['code','total_dollars']].head(20))
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
        
    #    unique_code_pivot_df[['code','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
        
        
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["spc"]).transform(sum)
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False]).copy()
        #print("pv=",pivot_df)
        name="Top 20 customers special price category by $purchases in the last 30 days"
        unique_code_pivot_df=pivot_df.drop_duplicates('spc',keep='first')
        unique_code_pivot_df.replace({'spc':dd2.spc_dict},inplace=True)
        unique_code_pivot_df.replace({'spc':dd2.spcs_dict},inplace=True)
       
        #unique_code_pivot_df=pd.unique(pivot_df['code'])
        print("\n",name)
        print(unique_code_pivot_df[['spc','total_dollars']].head(20))
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","-*")]=output_dir+name+".xlsx"
        
     #   unique_code_pivot_df[['specialpricecat','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
        
        
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["salesrep"]).transform(sum)
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False]).copy()
        #print("pv=",pivot_df)
        name="Top salesreps by $sales in the last 30 days"
        unique_code_pivot_df=pivot_df.drop_duplicates('salesrep',keep='first')
        unique_code_pivot_df.replace({'salesrep':dd2.salesrep_dict},inplace=True)
        
        #unique_code_pivot_df=pd.unique(pivot_df['code'])
        print("\n",name)
        print(unique_code_pivot_df[['salesrep','total_dollars']].head(20))
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","-*")]=output_dir+name+".xlsx"
        
        # unique_code_pivot_df[['salesrep','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
        
       # latest_date=sales_df['date'].max()
        
        end_date=all_sales_df['date'].iloc[-1]- pd.Timedelta(365, unit='d')
        year_sales_df=all_sales_df[all_sales_df['date']>end_date].copy()
    
        
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["salesrep"]).transform(sum)
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False]).copy()
        #print("pv=",pivot_df)
        name="Top salesreps by $sales in the last 365 days"
        unique_code_pivot_df=pivot_df.drop_duplicates('salesrep',keep='first')
        unique_code_pivot_df.replace({'salesrep':dd2.salesrep_dict},inplace=True)
        
        #unique_code_pivot_df=pd.unique(pivot_df['code'])
        print("\n",name)
        print(unique_code_pivot_df[['salesrep','total_dollars']].head(20))
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","-*")]=output_dir+name+".xlsx"
        
      #  unique_code_pivot_df[['salesrep','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
        
        
        
      #  latest_date=sales_df['date'].max()
        
        end_date=all_sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
        year_sales_df=all_sales_df[all_sales_df['date']>end_date].copy()
    
        
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["product"]).transform(sum)
        year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["product"]).transform(sum)
        
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
        #print("pv=",pivot_df)
        unique_code_pivot_df=pivot_df.drop_duplicates('product',keep='first')
        
        name="Top 50 products by $sales in the last 30 days"
        #unique_code_pivot_df=pd.unique(pivot_df['code'])
        print("\n",name)
        print(unique_code_pivot_df[['product','total_units','total_dollars']].head(20))
        #pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
        
      #  unique_code_pivot_df[['product','total_units','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
        
        
        
        
        year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["pg"]).transform(sum)
        year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["pg"]).transform(sum)
        pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False]).copy()
        unique_pg_pivot_df=pivot_df.drop_duplicates('pg',keep='first')
        unique_pg_pivot_df.replace({'pg':dd2.productgroup_dict},inplace=True)
        unique_pg_pivot_df.replace({'pg':dd2.productgroups_dict},inplace=True)
        
        name="Top productgroups by $sales in the last 30 days"
        print("\n",name)
        print(unique_pg_pivot_df[['pg','total_units','total_dollars']].head(20))
        #pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 
        # dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
        # dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
        
      #  unique_pg_pivot_df[['productgroup','total_units','total_dollars']].head(20).to_excel(output_dir+name+".xlsx") 
        
        
        name="Top 20 Credits in past 30 days"
        print("\n",name)
        end_date=all_sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
        #print(end_date)
        #print("ysdf=",sales_df)
        month_sales_df=all_sales_df[all_sales_df['date']>end_date].copy()
        #print("msdf=",month_sales_df)
        credit_df=month_sales_df[(month_sales_df['salesval']<-100) | (month_sales_df['qty']<-10)]
        #print(credit_df.columns)
        credit_df=credit_df.sort_values(by=["salesval"],ascending=[True])
        
        print(credit_df.tail(20)[['date','code','glset','qty','salesval']])
        # dd.report_dict[dd.report(name,3,"_*","_*")]=credit_df
        # dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
        
     #   credit_df[['date','code','glset','qty','salesval']].tail(50).to_excel(output_dir+name+".xlsx") 
        print("\n=====================================================================\n")

        return
    
    
    






   
class sales_plot_class(object):
    def __init__(self):   #,in4file):
    #    self.sales_plot_init="sales_plot_init called"
    #    print(self.sales_plot_init)
#        print("sales plot 
        pass 
  
        
    def load(self,infile):
        pass
       # print("load sales plot class=",infile)
        return("returned sales plot query load")     


    def preprocess(self,df,mat):
        # rename qolumns
       
       # df.rename(columns=rename_dict, inplace=True)
    
     #   df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
      #  df['date']=df.index
      #  df['mat']=df['salesval'].rolling(mat,axis=0).sum()
    #    print("rename collumsn")
        
       # df.rename(columns=rename_dict, inplace=True)
    
   #     df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
      #  df['date']=df.index
        df['mat']=df['salesval'].rolling(window=mat,axis=0).sum()
      #  df=df[(df['mat']>=0)]
    #    print("rename collumsn")
        return df
           


          
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
    
    
    def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
    


    def mat(self,df_dict,mat,latest_date,output_dir):
        print("plotting mat plot type=",df_dict.keys(),mat,latest_date.strftime('%d/%m/%Y'),output_dir)
        for k,v in df_dict.items():
      #      mat_df=v.copy()
            mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
 #           loffset = '7D'
 #           weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
 #           weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
            
            if mat_df.shape[0]>mat:
                mat_df=self.preprocess(mat_df,mat)
     #           df['mat']=df['salesval'].rolling(mat,axis=0).sum()
          #      df=df[(df['mat']>=0)]
     
           #     print("end mat preprocess=\n",df)
               # styles1 = ['b-','g:','r:']
                styles1 = ['b-']
              # styles1 = ['bs-','ro:','y^-']
                linewidths = 2  # [2, 1, 4]
                       
                fig, ax = pyplot.subplots()
                ax=mat_df.iloc[mat:][['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
             
                ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places
    
                ax.set_title("["+self._clean_up_name(str(k))+"] $ sales moving total "+str(mat)+" weeks @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
                ax.legend(title="",fontsize=6)
                ax.set_xlabel("",fontsize=6)
                ax.set_ylabel("",fontsize=6)
               # ax.yaxis.set_major_formatter('${x:1.0f}')
                ax.yaxis.set_tick_params(which='major', labelcolor='green',
                             labelleft=True, labelright=False)
                
                self._save_fig(self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
                plt.close()
        return   
 
    
    def mat_stacked_product(self,df_dict,mat,latest_date,output_dir):
        print("plotting mat plot type=",df_dict.keys(),mat,latest_date.strftime('%d/%m/%Y'),output_dir)
        for k,v in df_dict.items():
         #   df=self.preprocess(df_dict[k],mat)
    #        mat_df=v.copy()
            mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
  #          loffset = '7D'
  #          weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
  #          weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
            
            
            if mat_df.shape[0]>mat:
                mat_df=self.preprocess(mat_df,mat)
                styles1 = ['b-']
              # styles1 = ['bs-','ro:','y^-']
                linewidths = 2  # [2, 1, 4]
                       
                fig, ax = pyplot.subplots()
                ax=mat_df.iloc[mat:][['mat']].plot.area(stacked=True)       
                ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places
    
                ax.set_title("["+self.clean_up_name(str(k))+"] $ sales stacked moving total "+str(mat)+" weeks @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
                ax.legend(title="",fontsize=6)
                ax.set_xlabel("",fontsize=6)
                ax.set_ylabel("",fontsize=6)
               # ax.yaxis.set_major_formatter('${x:1.0f}')
                ax.yaxis.set_tick_params(which='major', labelcolor='green',
                             labelleft=True, labelright=False)
                
                self._save_fig(self._clean_up_name(str(k))+"_dollars_moving_total_stacked",output_dir)
                plt.close()
        return   
 

  
     
# =============================================================================
#      
#      
#    
#     def compare_customers(self,df_dict,mat,cust_list,latest_date,output_dir):
#         print("compare customer plot type=",cust_list,latest_date,output_dir,"\r")
# 
#         kcount=1
#         for k in df_dict.keys():
#             df=df_dict[k].copy() 
#
#   
#             # p_map(p_compare_plot,cust_list)
#             
# 
# 
#   #   def p_compare_plot(self,cust)
#             scaling_point_week_no=dd2.dash2_dict['sales']['plots']['scaling_point_week_no']
#             styles1 = ['r-',"b-","g-","k-","y-",'r-',"b-","g-","k-","y-",'r-',"b-","g-","k-","y-"]
#                     # styles1 = ['bs-','ro:','y^-']
#             linewidths = 2  # [2, 1, 4]
# 
#                      
#             pg_list=list(set(df['pg']))
#         #     print(k,"prod list=",prod_list)
#         #    
#             #kcount=1
#             pcount=1
#             for pg in pg_list:
#                 print(" ->>plotting product group [",pg,"] ",pcount,"/",len(pg_list)," of ",kcount,"/",len(df_dict.keys()),"queries                                                                                   ",end="\r",flush=True)
#                   #latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
#               #  valid_fig=True
#                 fig, ax = pyplot.subplots()
#                 fig.autofmt_xdate()
#                 
#                 start_point=[]
#         #        print("compare customers on plot prod=",prod,"\n",sales_df)
#             #       print("\n")   
#               #  if sales_df.shape[0]>0:
#                 t_count=0
#                 for cust in cust_list:
#                 #    print("\rCustomer dollar sales graphs:",t_count,"/",ctotrun,end="\r",flush=True)
#               #      print("customers to plot together",cust,"product",prod)
#                   #   if prod=="":
#                         #if dd.dash_verbose:
#                 #         print("compare customers on plot - customers to plot together",cust)
#                   #      cust_sales=sales_df[sales_df['code']==cust].copy()
#                   #  else:
#                         # if dd.dash_verbose:
#               #          print("product",prod,"-customers to plot together",cust)
#                     cust_sales=df[(df['code']==str(cust)) & (df['pg']==str(pg))].copy()
#                 #    cust_sales2=cust_sales.copy()
#               #          print("cust_sause=\n",cust_sales)
#                     if cust_sales.shape[0]>0: 
#                         
#                         cust_sales.set_index('date',inplace=True)
#                         
#                         cust_sales=cust_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
#                       #   print("cust_sause2=\n",cust_sales)
#             
#               #          cust_sales['mat']=cust_sales['salesval'].rolling(dd.mat2,axis=0).mean()
#                         cust_sales['mat']=cust_sales['salesval'].rolling(mat,axis=0).mean()
#             
#                         #print("cust_sause3=\n",cust_sales)
#             
#                       #   try:
#   #                          start_point.append(cust_sales['mat'].iloc[scaling_point_week_no])
#                         if cust_sales.shape[0]>scaling_point_week_no:
#                             start_point.append(cust_sales['mat'].iloc[scaling_point_week_no])
#                         else:
#                             start_point.append(10)
#                         #cust_sales.index = pd.to_datetime('period', format='%d-%m-%Y',exact=False)
#                  
#                         # styles1 = ['b-','g:','r:']
#                         
#                             cust_sales=cust_sales.iloc[scaling_point_week_no-1:,:]
#                         # else:
#                         #     start_point.append(pd.DataFrame([]))
#                       #  except:
#                       #      pass
#                       #      if valid_fig:
#                       #          print("not enough sales data",cust,prod)
#                       #      valid_fig=False    
#                       #  else: 
#                         #     valid_fig=True
#                           
# #                            cust_sales[['mat']].plot(grid=True,use_index=True,fontsize=8,style=styles1[t_count], lw=linewidths,ax=ax)
#                         try: 
#                             cust_sales[['mat']].plot(grid=True,use_index=True,fontsize=8,style=styles1[t_count], lw=linewidths,ax=ax)
#                         except:
#                             pass
#     #                           ax.set_title("["+self.clean_up_name(str(k))+"] "+str(cust)+" "+str(prod)+" $ sales stacked moving total "+str(mat)+" weeks @w/c:"+str(latest_date),fontsize= 7)
#                         ax.set_title("["+self._clean_up_name(str(k))+"] "+str(cust)+" pg:"+str(pg)+" dollars/week moving total comparison  "+str(mat)+" weeks @w/c:"+str(latest_date),fontsize= 7)
#   
#                         ax.legend(cust_list,title="",fontsize=8)
#                         ax.set_xlabel("",fontsize=8)
#                         
#                     else:
#                       #   print("cust sales1 compare 0")
#                         start_point.append(10)
#                   #  print("comparison_count=",t_count,"startpint",start_point)
#                   #   if t_count<len(cust_list):
#                     t_count+=1     
#                     
#             
#                 #ax.set_title("["+self.clean_up_name(str(k))+"] "+str(cust)+" "+str(prod)+" $ sales stacked moving total "+str(mat)+" weeks @w/c:"+str(latest_date),fontsize= 7)
#               #  ax.legend(cust_list,title="")
#               #  ax.set_xlabel("",fontsize=8)
#                 self._save_fig("cust_"+str(cust_list[0])+"_pg_"+str(pg)+"_1_together_dollars_moving_total",output_dir)
#                 plt.close()
#               #  print("start point",start_point) 
#                 scaling=[100/start_point[i] for i in range(0,len(start_point))]
#               #   print("scaling",scaling)
#                 
#               #   print("\n")
#               
#                 fig, ax = pyplot.subplots()
#                 fig.autofmt_xdate()
#              
#             
#             #    print("cust sales=\n",cust_sales)
#             
#                 t_count=0
#             #    valid_fig=True
#                 for cust in cust_list:
#                 #    print("\rCustomer dollar sales graphs:",t_count,"/",ctotrun,end="\r",flush=True)
#               #       print("customers to plot together",cust)
#                   #  if prod=="":
#                   #      cust_sales=sales_df[sales_df['code']==cust].copy()
#                   #  else:
#                     cust_sales=df[(df['code']==str(cust)) & (df['pg']==str(pg))].copy()
#                 #    cust_sales=cust_sales2.copy()
#               #       cust_sales=sales_df[sales_df['code']==cust].copy()
#                     if cust_sales.shape[0]>0: 
#                         cust_sales.set_index('date',inplace=True)
#                         
#                         cust_sales=cust_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
#                 
#                         cust_sales['mat']=cust_sales['salesval'].rolling(mat,axis=0).mean()
#                 
#                         cust_sales['scaled_mat']=cust_sales['mat']*scaling[t_count]
#                         #cust_sales.index = pd.to_datetime('period', format='%d-%m-%Y',exact=False)
#                  
#                         # styles1 = ['b-','g:','r:']
#                         # try:
#                         if cust_sales.shape[0]>scaling_point_week_no:
#                             cust_sales=cust_sales.iloc[scaling_point_week_no-1:,:]
#                             
#                         try:    
#                             ax=cust_sales[['scaled_mat']].plot(grid=True,use_index=True,fontsize=7,style=styles1[t_count], lw=linewidths,ax=ax)
#                         except:
#                             pass
# 
#                         ax.set_title("["+self._clean_up_name(str(k))+"] "+str(cust)+" "+str(pg)+" Scaled Sales/week moving total comparison "+str(mat)+" weeks @w/c:"+str(latest_date),fontsize= 7)
#                     else:
#                         pass
#                         # print("cust sales scale 0")
#                     #     except:
#                     #         pass
#                       #       valid_fig=False
#                             #print("not enough data2",cust,prod)
#                     #    else:    
#                         #cust_sales[['scaled_mat']].plot(grid=True,use_index=True,title=str(prod)+" Scaled Sales/week moving total comparison "+str(dd.mat2)+" weeks @w/c:"+str(latest_date),style=styles1[t_count], lw=linewidths,ax=ax)
#                       #        pass
#                  
#           #      print("scaled comparison t_count=",t_count,scaling)
#                   #  if t_count<len(cust_list):
#                     t_count+=1  
#                   
#                
#                 ax.legend(cust_list,title="",fontsize=8)
#                 ax.set_xlabel("",fontsize=8)
#             
#             
#             #    ax.axvline(dd.scaling_point_week_no, ls='--')
#             #
#              
#                 self._save_fig("cust_"+str(cust_list[0])+"_pg_"+str(pg)+"_2_scaled_together_dollars_moving_total",output_dir)
#                 plt.close()
#     #    print("cust sales2=\n",cust_sales,cust_sales.T)
#           
#                 pcount+=1
#           
#             
#             kcount+=1
#             
#         plt.close('all')    
#         print("\n")
#         return    
#         
#             
# =============================================================================
   
        
     
  
    def _p_compare_customers_plot(self,mp_para_list):
        scaling_point_week_no=dd2.dash2_dict['sales']['plots']['scaling_point_week_no']
        styles1 = ['r-',"b-","g-","k-","y-",'r-',"b-","g-","k-","y-",'r-',"b-","g-","k-","y-"]
                # styles1 = ['bs-','ro:','y^-']
        linewidths = 2  # [2, 1, 4]
   #     fig, ax = pyplot.subplots()
   #     fig.autofmt_xdate()
              
    #    start_point=[]

        query_name=mp_para_list[0]
        plot_df=mp_para_list[1]
        product_groups=mp_para_list[2]
        cust_list=mp_para_list[3]
        mat=mp_para_list[4]
        latest_date=mp_para_list[5]
        output_dir=mp_para_list[6]
      #  print("plot query name=",query_name)         
    
        for pg in product_groups:
            t_count=0
            start_point=[]
            fig, ax = pyplot.subplots()
            fig.autofmt_xdate()
  
            for cust in cust_list:
          #      print("cust=",cust,"pg=",pg)
                cust_sales=plot_df[(plot_df['code']==str(cust)) & (plot_df['pg']==str(pg))].copy()
                if cust_sales.shape[0]>0: 
                    cust_sales.set_index('date',inplace=True)
                    cust_sales=cust_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
                    
 #                  loffset = '7D'
          #          weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
           #         weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
                    
                    
                    cust_sales['mat']=cust_sales['salesval'].rolling(mat,axis=0).mean()
                    if cust_sales.shape[0]>scaling_point_week_no:
                        start_point.append(cust_sales['mat'].iloc[scaling_point_week_no])
                    else:
                        start_point.append(10)
                             
                    cust_sales=cust_sales.iloc[scaling_point_week_no-1:,:]
                    try: 
                        cust_sales[['mat']].plot(grid=True,use_index=True,fontsize=8,style=styles1[t_count], lw=linewidths,ax=ax)
                    except:
                        pass
                    ax.set_title("["+self._clean_up_name(str(query_name))+"] "+str(cust)+" pg:"+str(pg)+" dollars/week moving total comparison  "+str(mat)+" weeks @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
          
                #    ax.legend(cust,title="",fontsize=8)
                #    ax.set_xlabel("",fontsize=8)
                else:
                    start_point.append(10)
                    
                t_count+=1  
   
            ax.legend(cust_list,title="",fontsize=8)
            ax.set_xlabel("",fontsize=8)
     
            self._save_fig(self._clean_up_name(str(query_name))+"_cust_compare_"+str(cust)+"_pg_"+str(pg)+"_1_together_dollars_moving_total",output_dir)
            plt.close()
                        
            scaling=[100/start_point[i] for i in range(0,len(start_point))]
                  
            fig, ax = pyplot.subplots()
            fig.autofmt_xdate()
            
            t_count=0
            for cust in cust_list:
     
                cust_sales=plot_df[(plot_df['code']==str(cust)) & (plot_df['pg']==str(pg))].copy()
                if cust_sales.shape[0]>0: 
                   cust_sales.set_index('date',inplace=True)
                  
                   cust_sales=cust_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
                   
 #                  loffset = '7D'
  #                  weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
   #                  weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
                   
          
                   cust_sales['mat']=cust_sales['salesval'].rolling(mat,axis=0).mean()
          
                   cust_sales['scaled_mat']=cust_sales['mat']*scaling[t_count]
                  #cust_sales.index = pd.to_datetime('period', format='%d-%m-%Y',exact=False)
           
                  # styles1 = ['b-','g:','r:']
                  # try:
                   if cust_sales.shape[0]>scaling_point_week_no:
                       cust_sales=cust_sales.iloc[scaling_point_week_no-1:,:]
                      
                   try:    
                       ax=cust_sales[['scaled_mat']].plot(grid=True,use_index=True,fontsize=7,style=styles1[t_count], lw=linewidths,ax=ax)
                   except:
                       pass
        
                   ax.set_title("["+self._clean_up_name(str(query_name))+"] "+str(cust)+" pg:"+str(pg)+" Scaled Sales/week moving total comparison "+str(mat)+" weeks @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
                else:
                   pass
                t_count+=1  
            
         
            ax.legend(cust_list,title="",fontsize=8)
            ax.set_xlabel("",fontsize=8)
           
            self._save_fig(self._clean_up_name(str(query_name))+"_cust_compare_"+str(cust)+"_pg_"+str(pg)+"_2_scaled_together_dollars_moving_total",output_dir)
            plt.close()

    
      
     #   print("\n")
        return    


     
    def p_compare_customers(self,df_dict,mat,cust_list,latest_date,output_dir):
        print("compare customer plot type=",cust_list,latest_date.strftime('%d/%m/%Y'),output_dir,"\r")
     #   print("mp para list=",mp_para_list)
        mp_para_list=[]
        kcount=1
        for k,v in df_dict.items():
            mat_df=v.copy()                
            pg_list=list(set(mat_df['pg']))
            mp_para_list.append([k,mat_df,pg_list,cust_list,mat,latest_date,output_dir])
    #    print("mp para list=",mp_para_list)
        p_map(self._p_compare_customers_plot,mp_para_list)
        plt.close('all')   
        return            


         
    
        
    
    
  
    def _p_compare_products_plot(self,mp_para_list):
        scaling_point_week_no=dd2.dash2_dict['sales']['plots']['scaling_point_week_no']
        styles1 = ['r-',"b-","g-","k-","y-",'r-',"b-","g-","k-","y-",'r-',"b-","g-","k-","y-"]
                # styles1 = ['bs-','ro:','y^-']
        linewidths = 2  # [2, 1, 4]
   #     fig, ax = pyplot.subplots()
   #     fig.autofmt_xdate()
              
    #    start_point=[]

        query_name=mp_para_list[0]
        plot_df=mp_para_list[1]
        prod_list=mp_para_list[2]
        spc_list=mp_para_list[3]
        mat=mp_para_list[4]
        latest_date=mp_para_list[5]
        output_dir=mp_para_list[6]
      #  print("plot query name=",query_name)         
    
        for spc in spc_list:
            t_count=0
            start_point=[]
            fig, ax = pyplot.subplots()
            fig.autofmt_xdate()
  
            for prod in prod_list:
          #      print("cust=",cust,"pg=",pg)
                prod_sales=plot_df[(plot_df['product']==str(prod)) & (plot_df['spc']==spc)].copy()
                if prod_sales.shape[0]>0: 
                    prod_sales.set_index('date',inplace=True)
                    prod_sales=prod_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
                    
 #                   loffset = '7D'
  #                   weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
   #                   weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
                    
                    prod_sales['mat']=prod_sales['qty'].rolling(mat,axis=0).mean()
                    if prod_sales.shape[0]>scaling_point_week_no:
                        start_point.append(prod_sales['mat'].iloc[scaling_point_week_no])
                    else:
                        start_point.append(10)
                             
                    prod_sales=prod_sales.iloc[scaling_point_week_no-1:,:]
                    try: 
                        prod_sales[['mat']].plot(grid=True,use_index=True,fontsize=8,style=styles1[t_count], lw=linewidths,ax=ax)
                    except:
                        pass
                    ax.set_title("["+self._clean_up_name(str(query_name))+"] spc:"+str(spc)+" prod:"+str(prod)+" units/week moving total comparison  "+str(mat)+" weeks @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
          
                #    ax.legend(cust,title="",fontsize=8)
                #    ax.set_xlabel("",fontsize=8)
                else:
                    start_point.append(10)
                    
                t_count+=1  
   
            ax.legend(prod_list,title="",fontsize=8)
            ax.set_xlabel("",fontsize=8)
     
            self._save_fig(self._clean_up_name(str(query_name))+"_prod_compare_"+str(spc)+"_prod_"+str(prod)+"_3_together_units_moving_total",output_dir)
            plt.close()
                        
            scaling=[100/start_point[i] for i in range(0,len(start_point))]
                  
            fig, ax = pyplot.subplots()
            fig.autofmt_xdate()
            
            t_count=0
            for prod in prod_list:
     
                prod_sales=plot_df[(plot_df['product']==str(prod)) & (plot_df['spc']==spc)].copy()
                if prod_sales.shape[0]>0: 
                   prod_sales.set_index('date',inplace=True)
                  
                   prod_sales=prod_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
                   
              #    loffset = '7D'
              #    weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
              #    weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
                   
          
                   prod_sales['mat']=prod_sales['qty'].rolling(mat,axis=0).mean()
          
                   prod_sales['scaled_mat']=prod_sales['mat']*scaling[t_count]
                  #cust_sales.index = pd.to_datetime('period', format='%d-%m-%Y',exact=False)
           
                  # styles1 = ['b-','g:','r:']
                  # try:
                   if prod_sales.shape[0]>scaling_point_week_no:
                       prod_sales=prod_sales.iloc[scaling_point_week_no-1:,:]
                      
                   try:    
                       ax=prod_sales[['scaled_mat']].plot(grid=True,use_index=True,fontsize=7,style=styles1[t_count], lw=linewidths,ax=ax)
                   except:
                       pass
        
                   ax.set_title("["+self._clean_up_name(str(query_name))+"] spc:"+str(spc)+" prod:"+str(prod)+" Scaled units/week moving total comparison "+str(mat)+" weeks @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
                else:
                   pass
                t_count+=1  
            
         
            ax.legend(prod_list,title="",fontsize=8)
            ax.set_xlabel("",fontsize=8)
           
            self._save_fig(self._clean_up_name(str(query_name))+"_prod_compare_"+str(spc)+"_prod_"+str(prod)+"_4_scaled_together_units_moving_total",output_dir)
            plt.close()

    
      
     #   print("\n")
        return    


     
    def p_compare_products(self,df_dict,mat,prod_list,latest_date,output_dir):
        print("compare products plot type=",prod_list,latest_date.strftime('%d/%m/%Y'),output_dir,"\r")
     #   print("mp para list=",mp_para_list)
        mp_para_list=[]
        kcount=1
        for k,v in df_dict.items():
            mat_df=v.copy()                
            spc_list=list(set(mat_df['spc']))
       #     print("spclist=",spc_list)
            mp_para_list.append([k,mat_df,prod_list,spc_list,mat,latest_date,output_dir])
    #    print("mp para list=",mp_para_list)
        p_map(self._p_compare_products_plot,mp_para_list)     # unordered map
        plt.close('all')   
        return            

    
            

    def trend(self,df_dict):
        print("trend plot type=",df_dict)
        return("returned plot")


    def yoy_dollars(self,df_dict,mat,latest_date,output_dir):
        print("yoy dollars plot type=",mat,latest_date.strftime('%d/%m/%Y'),output_dir)
        for k,v in df_dict.items():
            cust_df=self.preprocess(v,mat)
            cust_df=cust_df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
            
            
  #          loffset = '7D'
  #          weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
  #          weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
            
            

          #  print("end yoy customer preprocess",k)
            left_y_axis_title="$/week"
        
                
            year_list = cust_df.index.year.to_list()
            week_list = cust_df.index.week.to_list()
            month_list = cust_df.index.month.to_list()
            
            cust_df['year'] = year_list   #prod_sales.index.year
            cust_df['week'] = week_list   #prod_sales.index.week
            cust_df['monthno']=month_list
            cust_df.reset_index(drop=True,inplace=True)
            cust_df.set_index('week',inplace=True)
            
            week_freq=4.3
            #print("prod sales3=\n",prod_sales)
            weekno_list=[str(y)+"-W"+str(w) for y,w in zip(year_list,week_list)]
            #print("weekno list=",weekno_list,len(weekno_list))
            cust_df['weekno']=weekno_list
            yest= [dt.datetime.strptime(str(w) + '-3', "%Y-W%W-%w") for w in weekno_list]    #wednesday
            
            #print("yest=",yest)
            cust_df['yest']=yest
            improved_labels = ['{}'.format(calendar.month_abbr[int(m)]) for m in list(np.arange(0,13))]
            fig, ax = pyplot.subplots()
            fig.autofmt_xdate()
          #  ax.ticklabel_format(style='plain')
            styles=["b-","r:","g:","m:","c:"]
            new_years=list(set(cust_df['year'].to_list()))
            #print("years=",years,"weels=",new_years)
            for y,i in zip(new_years[::-1],np.arange(0,len(new_years))):
                test_df=cust_df[cust_df['year']==y]
              #  print(y,test_df)
                fig=test_df[['salesval']].plot(use_index=True,grid=True,style=styles[i],xlabel="",ylabel=left_y_axis_title,ax=ax,fontsize=8)
             
            ax.legend(new_years[::-1],fontsize=8)   #9
            ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))   #,byweekday=mdates.SU)
#            ax.set_xticklabels([""]+improved_labels,fontsize=8)
            ax.set_xticklabels([""]+improved_labels,fontsize=7)

            ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places
            ax.set_title("["+self._clean_up_name(str(k))+"] Year on Year $ sales / week",fontsize= 7)
  
         #   ax.yaxis.set_major_formatter('${x:1.0f}')
            ax.yaxis.set_tick_params(which='major', labelcolor='green',
                         labelleft=True, labelright=False)

            self._save_fig(self._clean_up_name(str(k))+"_yoy_dollars_total",output_dir)
            plt.close()
        return   
 



    def yoy_units(self,df_dict,mat,latest_date,output_dir):
        print("yoy units plot type=",mat,latest_date.strftime('%d/%m/%Y'),output_dir)
        for k,v in df_dict.items():
            cust_df=self.preprocess(v,mat)
            cust_df=cust_df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)

  #          loffset = '7D'
  #          weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
  #          weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 



          #  print("end yoy customer preprocess",k)
            left_y_axis_title="units/week"
        
                
            year_list = cust_df.index.year.to_list()
            week_list = cust_df.index.week.to_list()
            month_list = cust_df.index.month.to_list()
            
            cust_df['year'] = year_list   #prod_sales.index.year
            cust_df['week'] = week_list   #prod_sales.index.week
            cust_df['monthno']=month_list
            cust_df.reset_index(drop=True,inplace=True)
            cust_df.set_index('week',inplace=True)
            
            week_freq=4.3
            #print("prod sales3=\n",prod_sales)
            weekno_list=[str(y)+"-W"+str(w) for y,w in zip(year_list,week_list)]
            #print("weekno list=",weekno_list,len(weekno_list))
            cust_df['weekno']=weekno_list
            yest= [dt.datetime.strptime(str(w) + '-3', "%Y-W%W-%w") for w in weekno_list]    #wednesday
            
            #print("yest=",yest)
            cust_df['yest']=yest
            improved_labels = ['{}'.format(calendar.month_abbr[int(m)]) for m in list(np.arange(0,13))]
            fig, ax = pyplot.subplots()
            fig.autofmt_xdate()
          #  ax.ticklabel_format(style='plain')
            styles=["b-","r:","g:","m:","c:"]
            new_years=list(set(cust_df['year'].to_list()))
            #print("years=",years,"weels=",new_years)
            for y,i in zip(new_years[::-1],np.arange(0,len(new_years))):
                test_df=cust_df[cust_df['year']==y]
              #  print(y,test_df)
                fig=test_df[['qty']].plot(use_index=True,grid=True,style=styles[i],xlabel="",ylabel=left_y_axis_title,ax=ax,fontsize=8)
             
            ax.legend(new_years[::-1],fontsize=8)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))   #,byweekday=mdates.SU)
            ax.set_xticklabels([""]+improved_labels,fontsize=7)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 0 decimal places
            ax.set_title("["+self._clean_up_name(str(k))+"] Year on Year units sales / week",fontsize= 7)
  
         #   ax.yaxis.set_major_formatter('${x:1.0f}')
            ax.yaxis.set_tick_params(which='major', labelcolor='green',
                         labelleft=True, labelright=False)

            self._save_fig(self._clean_up_name(str(k))+"_yoy_units_total",output_dir)
            plt.close()
        return   
 



    def pareto_customer(self,df_dict,latest_date,output_dir):
        print("pareto customer plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
   #     i_dict=df_dict.copy()
        for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
            new_df=v.groupby(['code'],sort=False).sum()
       #     print("pareto customer",k,new_df,new_df.shape)
        #    print("pareto customer",k,new_df)
            if new_df.shape[0]>0:
                new_df=new_df[(new_df['salesval']>1.0)]
                new_df=new_df[['salesval']].sort_values(by='salesval',ascending=False)   
            #    new_df=new_df.droplevel([0])
        
                new_df['ccount']=np.arange(1,new_df.shape[0]+1)
                df_len=new_df.shape[0]
                
                ptt=new_df['salesval']
                ptott=ptt.sum()
                new_df['cumulative']=np.cumsum(ptt)/ptott
                new_df=new_df.head(top)
                
                fig, ax = pyplot.subplots()
                fig.autofmt_xdate()
              #  ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places

#                ax.yaxis.set_major_formatter('${x:1.0f}')
              #  ax.yaxis.set_tick_params(which='major', labelcolor='green',
              #           labelleft=True, labelright=False)

             #   ax.ticklabel_format(style='plain') 
             #   ax.yaxis.set_major_formatter(ScalarFormatter())
          
                #ax.ticklabel_format(style='plain') 
          #      ax.axis([1, 10000, 1, 100000])
                
                ax=new_df.plot.bar(y='salesval',ylabel="",fontsize=7,grid=False)
            #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
           #     axis.set_major_formatter(ScalarFormatter())
             #   ax.ticklabel_format(style='plain')
                ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places

                ax.yaxis.set_tick_params(which='major', labelcolor='green',labelleft=True, labelright=False)
                ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" customer $ ranking total dollars "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
             
             
                ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=True,style=["r-"],secondary_y=True)
                ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
                ax3 = ax.twiny() 
                ax4=new_df[['ccount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
                if df_len<=1:
                    df_len=2
         
                
                ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
        
                self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_customer_$_ranking",output_dir)
                plt.close()
            else:
                print("pareto customer nothing plotted. no records for ",k,new_df)
 
        return


    def pareto_product_dollars(self,df_dict,latest_date,output_dir):
        print("pareto product plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
     #   i_dict=df_dict.copy()
        for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
            new_df=v.groupby(['product'],sort=False).sum()
 
      #      print("pareto product dollars",k,new_df,new_df.shape)
            if new_df.shape[0]>0:
                new_df=new_df[(new_df['salesval']>1.0)]
                new_df=new_df[['salesval']].sort_values(by='salesval',ascending=False)   
            #    new_df=new_df.droplevel([0])
        
                new_df['pcount']=np.arange(1,new_df.shape[0]+1)
                df_len=new_df.shape[0]
                
                ptt=new_df['salesval']
                ptott=ptt.sum()
                new_df['cumulative']=np.cumsum(ptt)/ptott
                new_df=new_df.head(top)
                
                fig, ax = pyplot.subplots()
                fig.autofmt_xdate()
 #               ax.yaxis.set_major_formatter('${x:1.0f}')
 
             #   ax.ticklabel_format(style='plain') 
             #   ax.yaxis.set_major_formatter(ScalarFormatter())
          
                #ax.ticklabel_format(style='plain') 
          #      ax.axis([1, 10000, 1, 100000])
                
                ax=new_df.plot.bar(y='salesval',ylabel="",fontsize=7,grid=False)
            #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
           #     axis.set_major_formatter(ScalarFormatter())
             #   ax.ticklabel_format(style='plain')
                ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places

                ax.yaxis.set_tick_params(which='major', labelcolor='green',labelleft=True, labelright=False)
                ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" product $ ranking total dollars "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
  
             
             
             
                ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=True,style=["r-"],secondary_y=True)
                ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
                ax3 = ax.twiny() 
                ax4=new_df[['pcount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
                if df_len<=1:
                    df_len=2
         
                
                ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
        
                self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_product_$_ranking",output_dir)
                plt.close()
            else:
                print("pareto product dollars nothing plotted. no records for ",k,new_df)
      
        return




    def pareto_product_units(self,df_dict,latest_date,output_dir):
        print("pareto product units plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
   #     i_dict=df_dict.copy()
      #  print("pareto product i_dict=\n",i_dict,"\n i_dict.items()=\n",i_dict.items())
        for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
            new_df=v.groupby(['product'],sort=False).sum()
 
       #     print("\n++++++pareto product units",k,new_df)
            if new_df.shape[0]>0:
                new_df=new_df[(new_df['qty']>1.0)]
                new_df=new_df[['qty']].sort_values(by='qty',ascending=False)   
            #    new_df=new_df.droplevel([0])
        
                new_df['pcount']=np.arange(1,new_df.shape[0]+1)
                df_len=new_df.shape[0]
                
                ptt=new_df['qty']
                ptott=ptt.sum()
                new_df['cumulative']=np.cumsum(ptt)/ptott
                new_df=new_df.head(top)
                
                fig, ax = pyplot.subplots()
                fig.autofmt_xdate()
 #               ax.yaxis.set_major_formatter('${x:1.0f}')
 
             #   ax.ticklabel_format(style='plain') 
             #   ax.yaxis.set_major_formatter(ScalarFormatter())
          
                #ax.ticklabel_format(style='plain') 
          #      ax.axis([1, 10000, 1, 100000])
                
                ax=new_df.plot.bar(y='qty',ylabel="units",fontsize=7,grid=False)
            #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
           #     axis.set_major_formatter(ScalarFormatter())
             #   ax.ticklabel_format(style='plain')
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:1.0f}')) # 0 decimal places

                ax.yaxis.set_tick_params(which='major', labelcolor='green',labelleft=True, labelright=False)
                ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" product unit ranking total units "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
  
                ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=True,style=["r-"],secondary_y=True)
                ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
                ax3 = ax.twiny() 
                ax4=new_df[['pcount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
                if df_len<=1:
                    df_len=2
         
                
                ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
        
                self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_product_units_ranking",output_dir)
                plt.close()
            else:
                print("pareto product units nothing plotted. no records for ",k,new_df)
        return

 

 


        
   
class sales_predict_class(object):
    def __init__(self):  #,in6file):
   #     self.sales_predict_init="sales_predict_init called"
   #     print(self.sales_predict_init)
        pass
        
        
        
        
        
        
            
          
    def log_dir(self,prefix=""):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "./dash2_outputs"
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)
      
            
                
   
        
        
        
  
          
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
        
        
        
        
        
   
    
       
    def _multiple_slice_scandata(self,df,query):
        new_df=df.copy()
        for q in query:
            
            criteria=q[1]
         #   print("key=",key)
         #   print("criteria=",criteria)
            ix = new_df.index.get_level_values(criteria).isin(q)
            new_df=new_df[ix]    #.loc[:,(slice(None),(criteria))]
        new_df=new_df.sort_index(level=['sortorder'],ascending=[True],sort_remaining=True)   #,axis=1)
    
      #  write_excel2(new_df,"testdf2.xlsx")
        return new_df
  
    
    
    
    
    
    
    def _write_excel(self,df,filename):
        sheet_name = 'Sheet1'
        writer = pd.ExcelWriter(filename,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')     
        df.to_excel(writer,sheet_name=sheet_name,header=False,index=False)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')     
        writer.save()
        return

       
    def _rfr_model(self,new_df):   #,latest_date,next_week):   
       
       # new_df=pd.read_pickle("prior_pred_new_df.pkl")
       # print("_rfr model new_df\n",new_df,"\n",new_df.T)
       # new_df.fillna(0,inplace=True)
        
        colnames=new_df.columns.get_level_values('colname').to_list()[::3]     
        plotnumbers=new_df.columns.get_level_values('plotnumber').to_list()[::3]        
        
      #  print("cn=",colnames)
      #  print("pln=",plotnumbers)
        
        # r=1
        # totalr=len(plotnumbers)
        # pred_dict={}
        # inv_dict={}
        X=np.array([])
        y=np.array([])
        
        
        for row,name in zip(plotnumbers,colnames):
           # print("row=",row)
         #   name=colnames[r]
            
            X_full=new_df.xs(['71',row],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
         #   print("X_full",X_full.shape)
         # original   X=np.concatenate((X,X_full[5:-3]),axis=0)
            X=np.concatenate((X,X_full[5:-3]),axis=0)
       
            y_full=new_df.xs(['79',row],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
          #  y=y_full[6:-2]  
         #   print("yfull sp",y_full.shape)
        
 #           y=np.concatenate((y,y_full[6:-2]),axis=0)
            y=np.concatenate((y,y_full[6:-2]),axis=0)

            
            
            
        print("\nFit random forest Regressor...")
        X=X.reshape(-1,1)
        
        
        
        forest_reg=RandomForestRegressor(n_estimators=300)
        
        forest_reg.fit(X,y)
         
        joblib.dump(forest_reg,dd2.dash2_dict['sales']['predictions']["save_dir"]+dd2.dash2_dict['sales']['predictions']["RFR_order_predict_model_savefile"])
        print("RFR complete...") 
       # if answer2=="y":
     
     #   return(joblib.load("RFR_order_predict_model.pkl"))
        return forest_reg      #,model     
           
    
    
    
    def _get_invoiced_sales_and_shift(self,new_df,sales_df):
        new_df=new_df.droplevel([3,5,6,7,12])
        new_df=self._multiple_slice_scandata(new_df,query=[('79','plottype3')]).copy()
        new_df=new_df.droplevel([2,4,5,6])
        new_df=new_df.T

        plotnumber=new_df.columns.get_level_values('plotnumber').to_list() 
        retailer=new_df.columns.get_level_values('retailer').to_list()   
        product=new_df.columns.get_level_values('product').to_list()     
     
        sales_df=sales_df.sort_index()
        
        retailer_sales=pd.DataFrame([])
        for n,r,p in zip(plotnumber,retailer,product):
            sdf=sales_df[['qty']][(sales_df['spc']==float(r)) & (sales_df['product']==p)]
 
 
#            weekly_sdf=sdf.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)   
               #     week ending Wed night 
  #          weekly_sdf=sdf.resample('W-TUE', label='left', loffset=pd.DateOffset(days=7)).sum().round(0)  
  
            loffset = '7D'
            weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
            weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
            w2=weekly_sdf.sort_index().copy()
            w2=w2.rename(columns={"qty":(n,r,p,79)})
            retailer_sales=pd.concat((retailer_sales,w2),axis=1)
        retailer_sales.fillna(0,inplace=True)    
      #  retailer_sales=retailer_sales.T
        retailer_sales.columns = pd.MultiIndex.from_tuples(retailer_sales.columns)
        retailer_sales.columns.names=["plotnumber","retailer","product","plottype3"]
        return retailer_sales   #.iloc[:,-1]
    
    
    
    def predict_order(self,scan_df,sales_df,latest_date,output_dir):    
      #  scan_df=pd.read_pickle(dd.scan_df_save)
      #  print("original scan_df=\n",scan_df)
      # new_df2=multiple_slice_scandata(scan_df,query=[('99','plottype')])
     #   print("predict order latest date=",latest_date.strftime('%d/%m/%Y'))
       #n print("plk new_df2=\n",new_df2)
      #  print(scan_df)
      #  scan_df=scan_df.T
        new_df=self._multiple_slice_scandata(scan_df,query=[('100','plottype2')]) #,('72','plottype3'),('71','plottype3'),('79','plottype3')])
        retailer_info_df=new_df.copy()
        
        new_df=new_df.droplevel([1,2,3,4,5,6,7,8])
    
        new_df=new_df.iloc[:,7:-1]
     #   print("new_df=\n",new_df)
    
        new_df*=1000
        new_df=new_df.astype(np.int32)
        saved_new_df=new_df.copy()
      #  print("plk new_df2=\n",new_df,"\n",new_df.T)
        new_df=new_df.drop('80', level='plottype3')
      #  saved_new_df=new_df.copy()
    
    #    print("pkl new_df=\n",new_df)  
      #  print("new_df=\n",new_df)
        
        
      #  saved_new_df=new_df.copy()
        new_df=new_df.T
        colnames=new_df.columns.get_level_values('colname').to_list()[::2]     
        plotnumbers=new_df.columns.get_level_values('plotnumber').to_list()[::2]  
        sortorder=new_df.columns.get_level_values('sortorder').to_list()[::2] 
      #  plottypethree=new_df.columns.get_level_values('plottype3').to_list()[::3]       
     #   print("colnames",colnames,len(colnames))
     #   print("plotnumbers",plotnumbers,len(plotnumbers))
     #   print("sortorder",sortorder,len(sortorder))
        
      
        new_df=new_df.droplevel([2],axis=1)
       
       # colnames=list(set(colnames))
       # plotnumbers=list(set(plotnumbers))
       # print("after set colnames",colnames,len(colnames))
       # print("after set plotnumbers",plotnumbers,len(plotnumbers))
       #  print("after set sortorders",plotnumbers,len(plotnumbers))
      
      #  print("plottypethree",plottypethree,len(plottypethree))
      
             #   newpred=np.concatenate((X_fill,X_full,pred))
    
        
        print("\n")
        for row,name in zip(plotnumbers,colnames):
            sales_corr=new_df.xs(row,level='plotnumber',drop_level=False,axis=1).corr(method='pearson')
            sales_corr=sales_corr.droplevel([0,1])
         #   print("sales corr",sales_corr.shape)
        #    if sales_corr.shape[1]>=3:
         #   shifted_vs_scanned_off_promo_corr=round(sales_corr.iloc[0,1],3)
            shifted_vs_scanned_corr=round(sales_corr.iloc[0,1],3)
    
          #  print(name,"-shifted vs scanned total sales correlation=",shifted_vs_scanned_corr)
        #    print(name,"-shifted vs scanned off promo correlation=",shifted_vs_scanned_off_promo_corr)
    
            #   print("Correlations:\n",sales_corr)
     
            # print("row=",row)
            new_df[:-3].xs(row,level='plotnumber',drop_level=False,axis=1).plot(xlabel="",ylabel="Units/week")
            plt.legend(title="Invoiced vs scan units total/wk correlation:("+str(shifted_vs_scanned_corr)+")",loc='best',fontsize=8,title_fontsize=8)
         #   plt.show()
            self._save_fig("pred_align_"+name,output_dir)
          #  plt.show()
        plt.close('all') 
    #n    new_df=new_df.T
     
    
         
    #    new_df=multiple_slice_scandata(new_df,query=[('100','plottype2')])
    #    print("new=df=\n",new_df,new_df.shape)
        print("\n")
        
       #     latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
    #    latest_date=sales_df['date'].max()  
        next_week=latest_date+ pd.offsets.Day(7)
     # 
      #  new_df=new_df.T
        new_df=saved_new_df.copy()
       # print("new saved df=\n",new_df)
     #   new_df=new_df.T
        new_df[next_week]=np.nan
        new_df=new_df.T
        
       # print("predict order new new_df=\n",new_df)
 
        units_invoiced_df=self._get_invoiced_sales_and_shift(retailer_info_df,sales_df)
       # print("predict order units invoiced",units_invoiced_df) 
        scan_inv=(units_invoiced_df/1000).copy()
        scan_inv.to_excel(output_dir+"units_invoiced.xlsx")
        
        
        #  new_df=new_df.iloc[:-1]
     #   new_df.to_pickle("prior_pred_new_df.pkl",protocol=-1)
    
    ###################################################3
    # train random forest model
        forest_reg=self._rfr_model(new_df)   #,latest_date,next_week)
# 
  #+++++++++++++++++++++++++++++++++++++++++++++++++      
          
        r=0
        totalr=len(plotnumbers)
        pred_dict={}
        sort_order_dict={}
        this_weeks_sales_dict={}
        inv_dict={}
        rfr_dict={}
       # rfr_list=[]
        extra_scan_inv=pd.DataFrame([]) 
       
        new_df=new_df.droplevel([2],axis=1)
      #  print("3 new_df=\n",new_df)
        for row,name in zip(plotnumbers,colnames):
        #    print("pred row=",row,"name=",name)
         #   name=colnames[r]
            
            X_full=new_df.xs(['71',row],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
            X=X_full[3:-1]
    #            X=new_df.iloc[:,7:-1].xs('1',level='plottype3',drop_level=False,axis=1).to_numpy()
      #      y_full2=new_df.xs(['79',row],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
            y_full=units_invoiced_df.xs([row],level=['plotnumber'],drop_level=False,axis=1).to_numpy().T[0]

    #        y=new_df.iloc[:,7:-1].xs('2',level='plottype3',drop_level=False,axis=1).to_numpy()
            y=y_full[3:-4]    #(dd2.dash2_dict['sales']['predictions']["invoiced_sales_weeks_offset"])]     
         #   print("pred",y[-3:],"d y_full",y_full)  #,"y_full2",y_full2) 
          #  new_df.replace(0,np.nan,inplace=True)
          
           # print(X,"Xshape",X.shape,"\n",y,"yshape",y.shape)
         #   print("old scan inv=\n",scan_inv,"\n")
       #     print("old scsn inv=\n",scan_inv)
 
            new_sd=pd.DataFrame({(row,99,name,'971'):pd.Series(X/1000),(row,99,name,'979'):pd.Series(y/1000),(row,"99","z_pred_"+name,'980'):pd.Series(np.zeros(y.shape[0]))})
        #    print("new_sd=\n",new_sd)
            extra_scan_inv=pd.concat([extra_scan_inv,new_sd], axis=1)   #ignore_index=True, axis=1)
            #scan_inv[row+"_X_"+name]=pd.Series(X/1000)
            #scan_inv[row+"_y_"+name]=pd.Series(y/1000)
            #scan_inv.sort_index(axis=1,level=0,inplace=True)
            #print(" sX=\n",pd.Series(X/1000))
         #   print("new scsn inv=\n",scan_inv)
            #pd.DataFrame({'email':sf.index, 'list':sf.values})
  
          
          
            old_preds=new_df.xs(['80',row],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
     
          #  old_preds[old_preds == 0] = np.nan # or use np.nan
         #   print("old preds=\n",old_preds,old_preds.shape)
       #     print("y_full=\n",y_full,y_full.shape)
          #  print(name)  #,"\nX=\n",X,X.shape,"\ny=\n",y,y.shape)
            
            # if answer2=="y":
            #     model=train_model(clean_up_name(str(name)),X,y,dd.batch_length,dd.no_of_batches,dd.epochs,r,totalr)
            #     pred=predict_order(X_full,y_full,name,model)
            #     pred_dict[name]=pred[0]
            # else:
            pred=np.array([-1])
            pred_dict[name]=-1
            #name_dict[name]=colnames[r]
            sort_order_dict[name]=sortorder[r]
     
            inv_dict[name]=y_full[-dd2.dash2_dict['sales']['predictions']["invoiced_sales_weeks_offset"]]   
            this_weeks_sales_dict[name]=y_full[-1]
         #   print(name,"predictions:",int(pred[0]))
          #  new_df=new_df.T
         #   print("level=",new_df.index.nlevels,"pred=",pred)
            lenneeded=new_df.shape[0]-len(y_full[:-1])-1
            if lenneeded>=0:
              #  y_fill=np.zeros(lenneeded)
                y_fill = np.empty(lenneeded)
                y_fill[:] = np.NaN
                newpred=np.concatenate((y_fill,y_full[:-1],[pred[0]]))
            else:
                newpred=np.concatenate((y_full[:-1],[pred[0]]))[-new_df.shape[0]:]
    
            
       #     print("newpred=\n",newpred,newpred.shape)
         #   new_df=new_df.T
         #   print("new df.T",new_df)
         #   new_df[(row,'73',name,'GRU_Prediction')]=newpred.astype(np.int32)
            
          #  new_df=new_df.T
            #new_df.iloc[:,-1]=pred[0]
           # new_df=new_df.T
         #   print("newdf2=\n",new_df)
            
  #          scan_inv["X_"+name]=pd.Series(X)
  #          scan_inv["y_"+name]=pd.Series(y)
  #          scan_inv.to_excel(output_dir+"units_invoiced_X_y.xlsx")
            rfr_pred=np.around(forest_reg.predict([[X[-1]]]),0) 
            
         #   rfrnewpred=np.concatenate((y_full[:-1],rfr_pred))[-new_df.shape[0]:]
            rfrnewpred=np.concatenate((old_preds[:-1],rfr_pred))[-new_df.shape[0]:]
            
            new_df[(row,'74',name,'RFR_Prediction')]=rfrnewpred.astype(np.int32)
     
    
    
          #  print("\nX[-1]=",X[-1],"DNN newpred[-1]=",newpred[-1],"vs Random forest pred=",rfr_pred)
            rfr_dict[name]=rfr_pred
           # rfr_list.append(rfr_pred)
     
    
            r+=1
            
       # extra_scan_inv.index=scan_inv.index[6:-1] 
        extra_scan_inv.index=scan_inv.index[6:-1]  
        extra_scan_inv=pd.concat([extra_scan_inv,scan_inv], axis=1) 
        extra_scan_inv.sort_index(axis=1,level=0,inplace=True)
        extra_scan_inv.columns.names=['plotnumber','retailer',"colname","plottype3"]
        extra_scan_inv.to_pickle(dd2.dash2_dict['sales']['predictions']["save_dir"]+dd2.dash2_dict['sales']['predictions']["units_invoiced_X_y_savefile"],protocol=-1)
        extra_scan_inv.to_excel(output_dir+dd2.dash2_dict['sales']['predictions']["units_invoiced_X_y"])
 
     #   print("final pred_dict=",pred_dict,"\ninv dict=",inv_dict)   
        pred_output_df=pd.DataFrame.from_dict(pred_dict,orient='index',columns=["GRU_order_prediction_"+str(next_week)],dtype=np.int32)
        sort_output_df=pd.DataFrame.from_dict(sort_order_dict,orient='index',columns=["sortorder"],dtype=np.int32)
      #  name_output_df=pd.DataFrame.from_dict(name_dict,orient='index',columns=["name"])
    
       #   sales resampled ending tuesdays, data drop is thursday so -2 day offset on dates
    
        inv_output_df=pd.DataFrame.from_dict(inv_dict,orient='index',columns=["invoiced_w/e_"+(latest_date+pd.offsets.Day(-2-7*(dd2.dash2_dict['sales']['predictions']["invoiced_sales_weeks_offset"]-1))).strftime("%d/%m/%Y")],dtype=np.int32)
        this_weeks_sales_output_df=pd.DataFrame.from_dict(this_weeks_sales_dict,orient='index',columns=["invoiced_w/e_"+(latest_date+pd.offsets.Day(-2)).strftime("%d/%m/%Y")],dtype=np.int32)
 
        rfr_output_df=pd.DataFrame.from_dict(rfr_dict,orient='index',columns=["RFR_order_prediction_"+(next_week+pd.offsets.Day(-2)).strftime("%d/%m/%Y")],dtype=np.int32)
     #   pred_output_df.replace(0.0,np.nan,inplace=True)
     #   pred_output_df=pd.concat((inv_output_df,pred_output_df,rfr_output_df),axis=1)
        pred_output_df=pd.concat((sort_output_df,this_weeks_sales_output_df,inv_output_df,pred_output_df,rfr_output_df),axis=1)
    
        #pred_output['invoiced_last_week']=new_df.xs('79',level='plottype3',drop_level=False,axis=1)[-1:].to_numpy().T[0]
    
      #  print("\nOrder predictions for next week (date is end of week)=\n",pred_output_df) #,"\n",pred_output_df.T)
        #print("\nRandom forest model predictions=",rfr_list)
    #    print("scan df=\n",scan_df)
        pred_output_df.drop(["GRU_order_prediction_"+str(next_week)],axis=1,inplace=True)
     #   pred_output_df.replace(np.nan,0,inplace=True)
      #  print("pred output=\n",pred_output_df)
        pred_output_df.sort_values(by="sortorder",ascending=True,inplace=True)
        pred_output_df.drop(["sortorder"],axis=1,inplace=True)
    
        print("\nafter Order predictions for next week (date is end of week)=\n",pred_output_df) #,"\n",pred_output_df.T)
       
       #     new_df=saved_new_df
      
       
        # #     results.to_pickle("order_predict_results.pkl")
        
       # pred_output_df=pred_output_df.T    
        sheet_name = 'Sheet1'
    
        writer = pd.ExcelWriter(output_dir+"order_predict_results.xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
       
        
        pred_output_df.to_excel(writer,sheet_name=sheet_name)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')
        
        writer.save()    
    
      #  new_df[(row,name,'prediction')]=X_full
     #   print("newdf=\n",new_df)
        
        new_df.sort_index(level=[0,1],axis=1,ascending=[True,True],inplace=True)
       # print("old new df=\n",new_df)
        new_df.replace(0,np.nan,inplace=True)
      #  print("new new_df=\n",new_df)
    #    new_df=new_df.droplevel(0)
        fig, ax = pyplot.subplots()
        fig.autofmt_xdate()
        #ax.ticklabel_format(style='plain')
        
     
        print("\nPlot order predictions for",next_week,"......")
        for row,name in zip(plotnumbers,colnames):
           # print("row=",row)
     
            new_df.iloc[-16:,:].xs(row,level='plotnumber',drop_level=False,axis=1).plot(xlabel="",sort_columns=True,style=['g:','r:','b-',"r-"],ylabel="Units/week")
        #    plt.autofmt_xdate()
     
            plt.legend(title="Invoiced units vs scanned units per week + next weeks prediction",loc='best',fontsize=6,title_fontsize=7)
            
        #    ax=plt.gca()
        #    ax.axhline(pred_dict[name], ls='--')
    #            ax2.axhline(30, ls='--')
    
          #  ax.text(1,1, "Next order prediction",fontsize=7)
     #           ax2.text(0.5,25, "Some text")
            self._save_fig("prediction_"+name,output_dir)
  
 
        print("Finished predict.\n\n\n")
        plt.close("all")
        return
  
    
 
    
    def _repredict_rfr(self,output_dir):
        print("\nRepredict all...")
        extra_scan_df=pd.read_pickle(dd2.dash2_dict['sales']['predictions']["save_dir"]+dd2.dash2_dict['sales']['predictions']["units_invoiced_X_y_savefile"])
      #  print("extra_scan inv=\n",extra_scan_df)
       # forest_reg=joblib.load(dd2.dash2_dict['sales']['predictions']["save_dir"]+dd2.dash2_dict['sales']['predictions']["RFR_order_predict_model_savefile"])
       # print("forest reg loaded",forest_reg)
     
        extra_scan_df=(extra_scan_df*1000).copy()
        
         #   print("X_full",X_full.shape)
         # original   X=np.concatenate((X,X_full[5:-3]),axis=0)
         #   X=np.concatenate((X,X_full[3:-1]),axis=0)
  
        plotnumbers=extra_scan_df.columns.get_level_values('plotnumber').to_list()[::4] 
        colnames=extra_scan_df.columns.get_level_values('colname').to_list()[::4]     

     #   print("pn=",plotnumbers,"cn=",colnames)
        rp_dict={}  
        for i in range(-20,0,1):
            
            X=np.array([])
            y=np.array([])
  
            for p in plotnumbers:
                X_full=extra_scan_df.iloc[20:i].xs(['971',p],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
                y_full=extra_scan_df.iloc[20:i].xs(['979',p],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
 
                X=np.concatenate((X,X_full),axis=0)
                y=np.concatenate((y,y_full),axis=0)

            X=X.reshape(-1,1)
        
            forest_reg=RandomForestRegressor(n_estimators=300)
          #  print("X,y=\n",X,y)
            forest_reg.fit(X,y)
  
    
          #  colnames=new_df.columns.get_level_values('colname').to_list()[::4]     
           # rp_dict={}
            for p,row in zip(plotnumbers,colnames):
                X=extra_scan_df.iloc[20:i].xs(['971',p],level=['plottype3','plotnumber'],drop_level=False,axis=1).to_numpy().T[0]
                rfr_pred=np.around(forest_reg.predict([[X[-1]]]),0)[0]      
               # print("pn=",p,row,"X shape=",X.shape,"[[X=[",i,"]]]=",[[X[-1]]],"rfr_pred=",rfr_pred)
                rp_dict[int(p),str(row),int(i)]=int(rfr_pred) 
     #   print(rp_dict)        
        rp_df=pd.DataFrame.from_dict(rp_dict,orient='index',columns=["prediction"])   #,index=extra_scan_df.index[-20:])  
       # rp_df.columns.names=["plotnumber","prediction_point"]
       # rp_df.index.names=["colname"]   #,"pred"]
       # rp_df=rp_df.T
        
        rp_df.index = pd.MultiIndex.from_tuples(rp_df.index,names=('plotnumber','colname','prediction_point'))
     #   print(rp_df)
        rp_df.sort_index(level=["plotnumber","prediction_point"],ascending=[True,True],inplace=True)
        print(rp_df.to_string())  
        rp_df.to_excel(output_dir+"units_invoiced_all_predictions.xlsx")
        print("\nRepredict finished.")
      #  prp_df=pd.pivot_table(rp_df,)
      #  print(prp_df)
        return     
  
    

    
  
    
  
    