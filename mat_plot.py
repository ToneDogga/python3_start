#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:06:10 2020

Created on Sat Jun 13 14:52:36 2020

@author: tonedogga
"""
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

import BB_data_dict as dd

import os
if dd.dash_verbose:
    pass
else:    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# TensorFlow ≥2.0 is required
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if len(gpus)>0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
if dd.dash_verbose==False:
     tf.autograph.set_verbosity(0,alsologtostdout=False)   
#     tf.get_logger().setLevel('INFO')
tf.config.experimental_run_functions_eagerly(False)   #True)   # false

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

#tf.autograph.set_verbosity(3, True)




from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"

 
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import timedelta
import calendar
import xlsxwriter

import xlrd

from pathlib import Path,WindowsPath
from random import randrange

import pickle
import multiprocessing

import warnings

from collections import namedtuple
from collections import defaultdict
from datetime import datetime
from pandas.plotting import scatter_matrix

from matplotlib import pyplot, dates
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.ticker as ticker


import time



#plt.ion() # enables interactive mode

#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)



def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./dashboard2_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)



output_dir = log_dir("dashboard2")
os.makedirs(output_dir, exist_ok=True)




# def load_sales(filenames):  # filenames is a list of xlsx files to load and sort by date
#     print("load:",filenames[0])
#     df=pd.read_excel(filenames[0],sheet_name="AttacheBI_sales_trans",usecols=range(0,17),verbose=False)  # -1 means all rows   
#  #   print("df size=",df.shape,df.columns)
#     for filename in filenames[1:]:
#         print("load:",filename)
#         new_df=pd.read_excel(filename,sheet_name="AttacheBI_sales_trans",usecols=range(0,17),verbose=False)  # -1 means all rows  
#         new_df['date'] = pd.to_datetime(new_df.date)  #, format='%d%m%Y')
#         print("appending",filename,":size=",new_df.shape)
#         df=df.append(new_df)
#         print("appended df size=",df.shape)
#    # +" w/c:("+str(latest_date)+")"
    
#     df.fillna(0,inplace=True)
    
#     #print(df)
#     print("drop duplicates")
#     df.drop_duplicates(keep='first', inplace=True)
#     print("after drop duplicates df size=",df.shape)
#     print("sort by date",df.shape[0],"records.\n")
#     df.sort_values(by=['date'], inplace=True, ascending=False)
      
#     #print(df.head(3))
#     #print(df.tail(3))
   
 
#     df["period"]=df.date.dt.to_period('D')
#  #   df["period"]=df.date.dt.to_period('W-THU')

#     df['period'] = df['period'].astype('category')
#   #  print("load sales df=\n",df)
    
 
#     return df           
 




def plot_type2(df,this_year_df,last_year_df):
    # first column is total units sales
    # second column is distribution 
    
    
   #3 print("plotdf.T=\n",df.T)
   # pv = pd.pivot_table(df.T, index=df.index, columns=df.index,
   #                 values='value', aggfunc='sum')
   # print("pv=\n",pv)
   # pv.plot()
   # plt.show()
      
    week_freq=8
   # print("plot type1 df=\n",df)
    this_year_df=this_year_df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
    last_year_df=last_year_df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
   
  #  df=df.T
  #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
  #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
  #  df=df.T
    this_year_df.iloc[:1]*=1000
    last_year_df.iloc[:1]*=1000

    #print("plot type1 df=\n",df)
    fig, ax = pyplot.subplots()
    
    
#    fig = plt.figure()
#ax1 = fig.add_subplot(111)
    ax2 = ax.twiny()



    fig.autofmt_xdate()
 
 #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('Units/week this year vs LY',fontsize=9)
    
    

    line=this_year_df.iloc[:1].T.plot(use_index=True,grid=True,xlabel="",kind='line',style=["r-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    line=last_year_df.iloc[:1].T.plot(use_index=True,grid=False,xlabel="",kind='line',style=["r:"],secondary_y=False,fontsize=9,legend=False,ax=ax2)   #,ax=ax2)

    if this_year_df.shape[0]>=2:
     #   line=last_year_df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b:'],secondary_y=False,fontsize=9,legend=False,ax=ax2)   #,ax=ax2)
        line=this_year_df.iloc[1:2].T.plot(use_index=True,grid=False,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
   # if df.shape[0]>=3:
   #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
#  ax.set_ylabel('Units/week',fontsize=9)

        ax.right_ax.set_ylabel('Distribution this year',fontsize=9)
    fig.legend(title="Units/week TY vs LY",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.4, 1.1))
  #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
  #  new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
  #  improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
  #  improved_labels=improved_labels[::week_freq]
  
  #  ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
  #  ax.set_xticklabels(improved_labels,fontsize=8)

    return


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(output_dir, fig_id + "." + fig_extension)
  #  print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
    return


def graph_sales_year_on_year(sales_df,title):
    prod_sales=sales_df[['salesval']].resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
    #print("prod sales1=\n",prod_sales)
    
    year_list = prod_sales.index.year.to_list()
    week_list = prod_sales.index.week.to_list()
    month_list = prod_sales.index.month.to_list()
    
    prod_sales['year'] = year_list   #prod_sales.index.year
    prod_sales['week'] = week_list   #prod_sales.index.week
    prod_sales['monthno']=month_list
    prod_sales.reset_index(drop=True,inplace=True)
    prod_sales.set_index('week',inplace=True)
    
    week_freq=4.3
    #print("prod sales3=\n",prod_sales)
    weekno_list=[str(y)+"-W"+str(w) for y,w in zip(year_list,week_list)]
    #print("weekno list=",weekno_list,len(weekno_list))
    prod_sales['weekno']=weekno_list
    yest= [dt.datetime.strptime(str(w) + '-3', "%Y-W%W-%w") for w in weekno_list]    #wednesday
    
    #print("yest=",yest)
    prod_sales['yest']=yest
    improved_labels = ['{}'.format(calendar.month_abbr[int(m)]) for m in list(np.arange(0,13))]
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
    ax.ticklabel_format(style='plain')
    
    new_years=list(set(prod_sales['year'].to_list()))
    #print("years=",years,"weels=",new_years)
    for y in new_years:
        test_df=prod_sales[prod_sales['year']==y]
      #  print(y,test_df)
        fig=test_df[['salesval']].plot(use_index=True,grid=True,xlabel="",ylabel="$",ax=ax,title=title,fontsize=8)
    
    
    ax.legend(new_years,fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))   #,byweekday=mdates.SU)
    ax.set_xticklabels([""]+improved_labels,fontsize=8)
    #  save_fig
    return




warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format
   
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors
visible_devices = tf.config.get_visible_devices('GPU') 

print("\nDash : Beerenberg TF2 Salestrans analyse/predict dashboard- By Anthony Paech 25/5/20")
print("=================================================================================================\n")       

if dd.dash_verbose:
     print("Python version:",sys.version)
     print("\ntensorflow:",tf.__version__)
     #    print("eager exec:",tf.executing_eagerly())      
     print("keras:",keras.__version__)
     print("numpy:",np.__version__)
     print("pandas:",pd.__version__)
     print("matplotlib:",mpl.__version__)      
     print("sklearn:",sklearn.__version__)         
     print("\nnumber of cpus : ", multiprocessing.cpu_count())            
     print("tf.config.get_visible_devices('GPU'):",visible_devices)
     print("\n=================================================================================================\n")       
    
     


np.random.seed(42)
tf.random.set_seed(42)
   
 ##############################################################################
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format

 
 
 ###################################################################################
 
#with open(dd.scan_dict_savename, 'rb') as g:
#    scan_dict = pickle.load(g)
 
#  scan_dict={"original_df":original_df,
#                "final_df":df,
sales_df=pd.read_pickle(dd.sales_df_savename)   #,protocol=-1)          
 
#sales_df=scan_dict['final_df']
#print(sales_df)

#sales_df=load_sales(dd.filenames)  # filenames is a list of xlsx files to load and sort by date
 #      with open(dd.sales_df_savename,"wb") as f:
 #            pickle.dump(sales_df, f,protocol=-1)
#sales_df.to_pickle(dd.sales_df_savename,protocol=-1)          
   
#   print("\n")   
sales_df.sort_values(by=['date'],ascending=True,inplace=True)
last_date=sales_df['date'].iloc[-1]
first_date=sales_df['date'].iloc[0]

print("Attache sales trans analysis up to date.  New save is:",dd.sales_df_savename)
print("Data available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")
#print(sales_df)
#dds=sales_df.groupby(['period'])['salesval'].sum().to_frame() 
#datelen=dds.shape[0]-365

#print("dds=\n",dds)    


sales_df['date']=pd.to_datetime(sales_df.date)
#sales_df['week']=pd.to_datetime(sales_df.date,format="%m-%Y")

#sales_df['year']=sales_df.date.year()
sales_df.set_index('date',inplace=True)
print("sales df1=\n",sales_df)   #,sales_df.T) 
graph_sales_year_on_year(sales_df,title="Total $ sales per week")
sales_df=sales_df[sales_df['code']=='FLPAS']
graph_sales_year_on_year(sales_df,title="FLPAS $ sales per week")


