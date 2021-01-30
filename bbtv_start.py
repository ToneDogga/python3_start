#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:30:24 2021

@author: tonedogga
"""
import string
import numbers
import time

import sys
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import dataframe_image as dfi

import uuid
import os
from pathlib import Path
from p_tqdm import p_map,p_umap
from PIL import Image, ImageOps

import datetime as dt
import calendar

import xlsxwriter

import xlrd
import joblib
import warnings
import pickle  

import pyodbc 
from datetime import datetime,timedelta

from shutil import copyfile


import pyglet
from pyglet import clock
from pyglet import gl
from pyglet.gl import *

from pyglet.window import key
from pyglet.window import mouse

from pyglet import shapes
from pyglet import image

from random import randrange

import matplotlib.pyplot as plt

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

from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay

import json   # for printing dictionaries
import multiprocessing  # for cpu count


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if len(gpus)>0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.autograph.set_verbosity(0, False)
import subprocess as sp

from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"

# =============================================================================
# if dd.dash_verbose==False:
#      tf.autograph.set_verbosity(0,alsologtostdout=False)   
#    #  tf.get_logger().setLevel('INFO')
# else:
#      tf.autograph.set_verbosity(1,alsologtostdout=True)   
# 
# =============================================================================



tf.config.run_functions_eagerly(False)
#os.chdir("/home/tonedogga/Documents/python_dev/bbtv1")

 
####################################################################################

import bbtv1_root
import bbtv1_dict as dd2


bbtv1=bbtv1_root.bbtv1_class()  
#os.chdir("/home/tonedogga/Documents/python_dev/bbtv1")

#############################################################################################

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format

#  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

visible_devices = tf.config.get_visible_devices('GPU') 


# if ODBC fails
load_file="raw_savefile.pkl"
load_dir="./bbtv1_saves/"

load_production_dir="./"
load_production_file="production_df.pkl"

  

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# def log_dir(prefix=""):
#     now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#     root_logdir = "./bbtv1_outputs"
#     if prefix:
#         prefix += "-"
#     name = prefix + "run-" + now
#     return "{}/{}/".format(root_logdir, name)

# #output_dir = log_dir("dashboard")


# os.chdir("/home/tonedogga/Documents/python_dev/bbtv1")
cwdpath = os.getcwd()
# print("cp",cwdpath)






# plot_output_dir = log_dir("bbtv1")
# dd2.bbtv_dict['sales']['plot_output_dir'] =plot_output_dir

# print("bod",plot_output_dir)


#  #print("\nLoading sales_df.pkl.....")
# #st=sales_trans()

 
# warnings.filterwarnings('ignore')       



# MY_DEBUG=False #True   #False
# CHAR_HEIGHT=19

# #window_objects={}

# keys = key.KeyStateHandler()
  
# batch = pyglet.graphics.Batch()
# canvas = {}


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# import pyglet
# from pyglet.gl import *

# from time import time # Used for FPS calc

# key = pyglet.window.key

# class main(pyglet.window.Window):
#     def __init__ (self):
#         super(main, self).__init__(800, 800, fullscreen = False, vsync = True)

#         self.running = True

#         self.fps_counter = 0
#         self.last_fps = time()
#         self.fps_text = pyglet.text.Label(str(self.fps_counter), font_size=12, x=10, y=10)

#     def on_key_press(self, symbol, modifiers):
#         if symbol == key.ESCAPE: # [ESC]
#             self.running = False
#         else:
#             self.clear() # However, this is done in the render() logic anyway.

#     def on_draw(self):
#         self.render()

#     def render(self):
#         self.clear()

#         # And flip the GL buffer
#         self.fps_counter += 1
#         if time() - self.last_fps > 1:
#             self.fps_text.text = str(self.fps_counter)
#             self.fps_counter = 0
#             self.last_fps = time()

#         self.fps_text.draw()

#         self.flip()

#     def run(self):
#         while self.running is True:
#             self.render()

#             # -----------> This is key <----------
#             # This is what replaces pyglet.app.run()
#             # but is required for the GUI to not freeze
#             #
#             event = self.dispatch_events()
#             if event and event.type == pygame.QUIT:
#                 self.running = False

# x = main()
# x.run()


         
    
    # def _save_fig(self,fig_id, output_dir,transparent,facecolor,tight_layout=True, fig_extension="png", resolution=300):
    #     os.makedirs(output_dir, exist_ok=True)
    #     path = os.path.join(output_dir, fig_id + "." + fig_extension)
    #   #  print("Saving figure", fig_id)
    #     if tight_layout:
    #         plt.tight_layout()
    #     plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight',transparent=transparent,facecolor=facecolor, edgecolor='none')
    #     return
     
  





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def display_start_banner(plot_output_dir):
       
    print("\nBBTV1-Beerenberg analyse, visualise and predict- By Anthony Paech 26/1/2021")
    print("=================================================================================================\n")       
    
    print("Python version:",sys.version)
    print("Current working directory",cwdpath)
    print("plot_output dir",plot_output_dir)
    print("data save directory",dd2.bbtv_dict['sales']['save_dir'])
 #   print("data save directory",dd2.bbtv_dict['sales']['plot_output_dir'])
    print("\ntensorflow:",tf.__version__)
    #print("TF2 eager exec:",tf.executing_eagerly())      
    print("keras:",keras.__version__)
    print("numpy:",np.__version__)
    print("pandas:",pd.__version__)
    print("pyglet:",pyglet.version)
    print("pyodbc:",pyodbc.version)
    print("matplotlib:",mpl.__version__)      
    print("sklearn:",sklearn.__version__)         
    print("\nnumber of cpus : ", multiprocessing.cpu_count())            
    print("tf.config.get_visible_devices('GPU'):\n",visible_devices)
 #   print("\n",pd.versions(),"\n")
    print("\n=================================================================================================\n")     

#os.chdir("/home/tonedogga/Documents/python_dev")
 #print("\nLoading sales_df.pkl.....")
#st=sales_trans()

 
#warnings.filterwarnings('ignore')       

#global window_objects
# window_objects={}
# window_objects["results"]={"object_type":"text_box","winobject":bbtv1.frontend.text_box(startx=0,starty=0,width=2300,height=300)}
# window_objects["working"]={"object_type":"text_box","winobject":bbtv1.frontend.text_box(startx=2300,starty=0,width=700,height=1800)}
# window_objects["instructions"]={"object_type":"text_box","winobject":bbtv1.frontend.text_box(startx=0,starty=1800,width=600,height=200)}
# window_objects["query_queue"]={"object_type":"text_box","winobject":bbtv1.frontend.text_box(startx=600,starty=1800,width=2400,height=200)}

# window_objects["results"]["winobject"].add_text("Results window ")
# window_objects["working"]["winobject"].add_text("Working window ")
# window_objects["instructions"]["winobject"].add_text("Instructions ")
# window_objects["query_queue"]["winobject"].add_text("Query queue window \n")




#window_objects=dict()


class SABusinessCalendar(AbstractHolidayCalendar):
   rules = [
     Holiday('New Year', month=1, day=1), #observance=sunday_to_monday),
     Holiday('Australia Day', month=1, day=26, observance=sunday_to_monday),
   
     Holiday('Anzac Day', month=4, day=25),
     Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
     Holiday('Easter Monday', month=1, day=1, offset=[Easter(),Day(+1)]),    

   
     Holiday('October long weekend', month=10, day=1, observance=sunday_to_monday),
     Holiday('Christmas', month=12, day=25, observance=nearest_workday),
     Holiday('Boxing day', month=12, day=26, observance=nearest_workday),
     Holiday('Dec27 shutdown', month=12, day=27),   #, observance=nearest_workday)    

     Holiday('Proclamation day', month=12, day=28, observance=nearest_workday),
     Holiday('Dec29 shutdown', month=12, day=29),   #, observance=nearest_workday)    
     Holiday('Dec30 shutdown', month=12, day=30),   #, observance=nearest_workday)     
     Holiday('Dec31 shutdown', month=12, day=31)   #, observance=nearest_workday)     

   ]
      




def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./bbtv_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

#output_dir = log_dir("dashboard")





def main():
  #  global window_objects
    
    os.chdir("/home/tonedogga/Documents/python_dev/bbtv1")
#    tmp=sp.call('clear',shell=True)  # clear screen 'use 'clear for unix, cls for windows
    plot_output_dir = log_dir("bbtv")
    dd2.bbtv_dict['sales']['plot_output_dir'] =plot_output_dir

    work_queue=[]
    SA_BD = CustomBusinessDay(calendar=SABusinessCalendar())
   # sales_df=st.load_pickle(load_dir,load_file)
    bbtv1=bbtv1_root.bbtv1_class()  

    display_start_banner(plot_output_dir)
    
    sales_df,stock_levels_df=bbtv1.firehose.load_sales_df(load_dir,load_file)
      
   # window_objects["working"]["winobject"].add_text("\n"+status_report+"\n")
    
    production_df,schedule_df,recipe_df=bbtv1.firehose.load_production_df(load_production_dir,load_production_file)
   # window_objects["working"]["winobject"].add_text("\n"+status_report2+"\n")
   # window_objects["results"]["winobject"].add_text(status_report3)
 
    
    work_queue=bbtv1.frontend.gui_queries(sales_df)

    start_timer = time.time()
    print("\nBBTV batched query queue started:",dt.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S %d/%m/%Y'))
    #bbtv1.sales.pivot.distribution_report_dollars('last_today_to_365_days',query_dict['last_today_to_365_days'],plot_output_dir,trend=True)
      
    #bbtv1.scheduler.animate_plots(gif_duration=4,mp4_fps=10,plot_output_dir=plot_output_dir)
 
  #  query_dict2=bbtv1.sales.query.queries(dd2.bbtv_dict['sales']['sales_df'])    
  
    
 #   bbtv1.animate.animate_brand_index(plot_output_dir+dd2.bbtv_dict['sales']['plots']["animation_plot_dump_dir"],plot_output_dir,mp4_fps=11)
   # bbtv1.animate.animate_query_dict(query_dict2,plot_output_dir+dd2.bbtv_dict['sales']['plots']["animation_plot_dump_dir"],plot_output_dir,mp4_fps=11)
 
    print(work_queue)       
  
    #bbtv1.ave_predict.actual_vs_expected(query_dict,plot_output_dir)
   


 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
   
    end_timer = time.time()
    print("\nBBTV batched query queue finished:",dt.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S %d/%m/%Y'))
    print("BBTV total queue runtime:",round(end_timer - start_timer,2),"seconds.\n")


       
       
    return

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    
   
main()   




# print(window_objects)
# for k,v in window_objects.items():
#     print(k)
#     for i,j in v.items():
#         print("    ",i,":",j)
        
        
# for k,v in window_objects.items():
#     print(k,v['object_name'])
    
    
# for v in window_objects.values():
#     print(v,v['object_name'])
     
#     # for i,j in v.items():
#      #   print("    ",i,":",j)

# print(window_objects['results']['object'])
# window_objects["results"]["winobject"].add_text("textingd")

