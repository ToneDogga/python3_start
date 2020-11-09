#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:51:34 2020

@author: tonedogga
"""

import subprocess as sp
tmp=sp.call('clear',shell=True)  # clear screen 'use 'clear for unix, cls for windows
 
import pandas as pd
import numpy as np

import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


import os

# TensorFlow ≥2.0 is required
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.run_functions_eagerly(False)
tf.autograph.set_verbosity(0, False)




from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"



# import numpy as np
# import pandas as pd
import datetime as dt
# from datetime import date
# from datetime import timedelta
import calendar
# import xlsxwriter

# import xlrd

# from pathlib import Path,WindowsPath
# from random import randrange

# import pickle
import multiprocessing

import warnings


#import subprocess as sp

#from collections import Counter
#from statistics import mean
# from multiprocessing import Pool, Process, Lock, freeze_support, Queue, Lock, Manager
#from tqdm import *
from p_tqdm import p_map

# from os import getpid
# #import os
# #import hashlib
# #import time
# #import pickle
# #import multiprocessing 




# from collections import namedtuple
# from collections import defaultdict
from datetime import datetime
# from pandas.plotting import scatter_matrix

from matplotlib import pyplot, dates
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

# import time
# import joblib
    

# import sklearn.linear_model
# import sklearn.neighbors

# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# from pandas.plotting import scatter_matrix


import salestrans_lib  
import query_dict as qd
import pyglet




   
def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./salestrans_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


output_dir = log_dir("salestrans_outputs")
os.makedirs(output_dir, exist_ok=True)

st=salestrans_lib.salestrans_df(output_dir)   # instantiate a salestrans_df




def graph_sales_year_on_year(df,title,left_y_axis_title):
  #  prod_sales=sales_df[['salesval']].resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
    #print("prod sales1=\n",prod_sales)
    if df.shape[0]>0:
        df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        df['qty_mat']=df['qty'].rolling(qd.smoothing_mat,axis=0).mean()
        df['salesval_mat']=df['salesval'].rolling(qd.smoothing_mat,axis=0).mean()
 
 
        year_list = df.index.year.to_list()
        week_list = df.index.week.to_list()
        month_list = df.index.month.to_list()
        
        df['year'] = year_list   #prod_sales.index.year
        df['week'] = week_list   #prod_sales.index.week
        df['monthno']=month_list
        df.reset_index(drop=True,inplace=True)
        df.set_index('week',inplace=True)
        
        week_freq=4.3
        #print("prod sales3=\n",prod_sales)
        weekno_list=[str(y)+"-W"+str(w) for y,w in zip(year_list,week_list)]
        #print("weekno list=",weekno_list,len(weekno_list))
        df['weekno']=weekno_list
        yest= [dt.datetime.strptime(str(w) + '-3', "%Y-W%W-%w") for w in weekno_list]    #wednesday
        
        #print("yest=",yest)
        df['yest']=yest
        improved_labels = ['{}'.format(calendar.month_abbr[int(m)]) for m in list(np.arange(0,13))]
        fig, ax = pyplot.subplots()
        fig.autofmt_xdate()
        ax.ticklabel_format(style='plain')
        styles=["b-","r:","g:","m:","c:"]
        new_years=list(set(df['year'].to_list()))
        #print("years=",years,"weels=",new_years)
        for y,i in zip(new_years[::-1],np.arange(0,len(new_years))):
            test_df=df[df['year']==y]
          #  print(y,test_df)
            fig=test_df[['salesval_mat']].plot(use_index=True,grid=True,style=styles[i],xlabel="",ylabel=left_y_axis_title,ax=ax,title=title,fontsize=8)
         
        ax.legend(new_years[::-1],fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))   #,byweekday=mdates.SU)
        ax.set_xticklabels([""]+improved_labels,fontsize=8)
        figname=title
        save_fig(figname)
     
    #  save_fig
    return
   


def pareto_product(df,title):
    top=60
    if df.shape[0]>0:
 
       # df=df.droplevel([0])
        df=df.groupby(['product'],sort=False).sum()
        df=df.sort_values(by='salesval',ascending=False)  
     #   print("pareto df=\n",df)
         
        ptt=df['salesval']
        ptott=ptt.sum()
        df['cumulative']=np.cumsum(ptt)/ptott
        df=df.head(top)
              
        fig, ax = pyplot.subplots()
        fig.autofmt_xdate()
        ax.ticklabel_format(style='plain')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        #ax.ticklabel_format(style='plain')
        
        ax=df.plot.bar(y='salesval',ylabel="$",fontsize=5,grid=True,title="Top "+str(top)+" product $ ranking for "+str(title))
    #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
   #    axis.set_major_formatter(ScalarFormatter())
     #   ax.ticklabel_format(style='plain') 
 
        ax2=df.plot(y='cumulative',xlabel="",rot=90,ax=ax,style=["r-"],secondary_y=True)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))

        save_fig("pareto_top_"+str(top)+"_product $ ranking for "+str(title))
        plt.close('all')

    return




def pareto_customer(df,title):
    top=60
    if df.shape[0]>0:
 
       # df=df.droplevel([0])
        df=df.groupby(['code'],sort=False).sum()
        df=df.sort_values(by='salesval',ascending=False)  
    #    print("pareto df=\n",df)
         
        ptt=df['salesval']
        ptott=ptt.sum()
        df['cumulative']=np.cumsum(ptt)/ptott
        df=df.head(top)
              
        fig, ax = pyplot.subplots()
        fig.autofmt_xdate()
        ax.ticklabel_format(style='plain')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        #ax.ticklabel_format(style='plain')
        
        ax=df.plot.bar(y='salesval',ylabel="$",fontsize=5,grid=True,title="Top "+str(top)+" customer $ ranking for "+str(title))
    #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
   #    axis.set_major_formatter(ScalarFormatter())
     #   ax.ticklabel_format(style='plain') 
 
        ax2=df.plot(y='cumulative',xlabel="",rot=90,ax=ax,style=["r-"],secondary_y=True)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))

        save_fig("pareto_top_"+str(top)+"_customer $ ranking for "+str(title))
        plt.close('all')

    return






def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(output_dir, fig_id + "." + fig_extension)
  #  print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
    return




# def smooth(df):
#  #    df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
#      df['qty_mat']=df['qty'].rolling(qd.smoothing_mat,axis=0).mean()
#      df['salesval_mat']=df['salesval'].rolling(qd.smoothing_mat,axis=0).mean()
 
#      return df



def build_an_entry(query_name):
  #  query_name=qd.queries[q]
    new_df=df.copy()
    for qn in query_name:  
        q_df=st.query_df(new_df,qn)
        new_df=q_df.copy()
    q_df.drop_duplicates(keep="first",inplace=True)    
   # q_df=smooth(q_df)
    return st.save_query(q_df,query_name,root=False)   
    


def build_query_dict(df):
    if df.shape[0]>0:
        df=df.rename(columns=qd.rename_columns_dict)  

    query_handles=[]
    query_handles.append(p_map(build_an_entry,qd.queries.values()))   #st.save_query(q_df,query_name,root=False)   
    return {k: v for k, v in zip(qd.queries.keys(),query_handles[0])}



def start_banner():
    
    visible_devices = tf.config.get_visible_devices('GPU') 

    print("\nSalestrans visual dashboard3- By Anthony Paech 7/11/20")
    print("=================================================================================================\n")       

    print("Python version:",sys.version)
    print("\ntensorflow:",tf.__version__)
    #    print("eager exec:",tf.executing_eagerly())      
    print("keras:",keras.__version__)
    print("numpy:",np.__version__)
    print("pandas:",pd.__version__)
    print("matplotlib:",mpl.__version__)      
    print("sklearn:",sklearn.__version__)         
    print("\nnumber of cpus : ", multiprocessing.cpu_count())            
    print("tf.config.get_visible_devices('GPU'):\n",visible_devices)
    
    print("\n=================================================================================================\n")       




def main():
 
 #   output_dir = st.log_dir("salestrans_outputs")
 #   os.makedirs(output_dir, exist_ok=True)


  #  global oneyear_sales_df,latest_date,lastoneyear_sales_df,twoyear_sales_df
    
  #  tmp=sp.call('clear',shell=True)  # clear screen 'use 'clear for unix, cls for windows
    start_banner()
   
    warnings.filterwarnings('ignore')
    pd.options.display.float_format = '{:.4f}'.format
      
  #  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors
    
    renew=(input("renew salestrans? (y/n)")=='y')  
    
    global df   # so I can use p_map multiprocessing  
    df=st.load(qd.sales_trans_filenames,renew=renew)  #=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"])
   # print(df)
    
   # print(df)
   # st.display_df(df) 
  
    ####################################################################################
    print("Build query dict\n")
    query_handles=build_query_dict(df)   
  #  print("query handles=",query_handles)    
  ##############################################################
  
    for q in query_handles.keys():
  #      print("qh=",qh)
        new_df,new_query_name=st.load_query(query_handles[q],root=False)
        print(q,"(",len(query_handles[q]),")=",new_query_name,"\n",new_df.shape,"\n")
        pareto_product(new_df,q)
        pareto_customer(new_df,q)
        graph_sales_year_on_year(new_df,q,"$/week")
 

 #   print(st.load_query(query_handles['not shop'],root=False)) 
 #   print(st.load_query(query_handles['online'],root=False)) 
   
    
            
main()

