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
# import datetime as dt
# from datetime import date
# from datetime import timedelta
# import calendar
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

# from matplotlib import pyplot, dates
# import matplotlib.dates as mdates
import matplotlib as mpl
# from matplotlib.pyplot import plot, draw, ion, show
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import seaborn as sns
# import matplotlib.ticker as ticker
# from matplotlib.ticker import ScalarFormatter

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


def build_an_entry(query_name):
  #  query_name=qd.queries[q]
    new_df=df.copy()
    for qn in query_name:  
        q_df=st.query_df(new_df,qn)
        new_df=q_df.copy()
    q_df.drop_duplicates(keep="first",inplace=True)       
    return st.save_query(q_df,query_name,root=False)   
    


def build_query_dict(df):
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
    
    global df   # so I can use p_map multiprocessing  
    df=st.load(qd.sales_trans_filenames,renew=False)  #=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"])
   # st.display_df(df) 
  
    ####################################################################################
    print("Build query dict\n")
    query_handles=build_query_dict(df)   
 #   print("query handles=",query_handles)    
  ##############################################################
  
    for q in query_handles.keys():
  #      print("qh=",qh)
        new_df,new_query_name=st.load_query(query_handles[q],root=False)
        print(q,"(",len(query_handles[q]),")=",new_query_name,new_df.shape,"\n")

    print(st.load_query(query_handles['not shop'],root=False)) 
    print(st.load_query(query_handles['online'],root=False)) 
   
    
            
main()

