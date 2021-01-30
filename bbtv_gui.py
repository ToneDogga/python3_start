#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:32:21 2021

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


 
####################################################################################

#os.chdir("/home/tonedogga/Documents/python_dev/bbtv1")
cwdpath = os.getcwd()
#print("cp",cwdpath)

#import bbtv1_root
import bbtv1_dict as dd2


#bbtv1=bbtv1_root.bbtv1_class()   #"in_dash value")   # instantiate a salestrans_df

#############################################################################################

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format

#  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

visible_devices = tf.config.get_visible_devices('GPU') 



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./bbtv1_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

#output_dir = log_dir("dashboard")






#os.chdir("/home/tonedogga/Documents/python_dev")
plot_output_dir = log_dir("bbtv1")
dd2.bbtv_dict['sales']['plot_output_dir'] =plot_output_dir

#print("bod",plot_output_dir)


 #print("\nLoading sales_df.pkl.....")
#st=sales_trans()

 
warnings.filterwarnings('ignore')       



MY_DEBUG=False #True   #False
CHAR_HEIGHT=19

#window_objects={}

keys = key.KeyStateHandler()
  
batch = pyglet.graphics.Batch()
canvas = {}



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# import pyglet
# from pyglet import clock
# from pyglet import gl
# from pyglet.gl import *

# from pyglet.window import key
# from pyglet.window import mouse

# from pyglet import shapes
# from pyglet import image

# import pandas as pd

# #from bbtv_init_test import button, buttons



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



class button(object):
    def __init__(self,*,window,name,sub_name,x_start,x_len,y_start,y_len,colour,pushed_colour,button_type,title,sub_title,active,visible,movable,pushable,floating,pushed,toggle,mouse_over,contains_list,unique_list,unique_list_start_point,allow_multiple_selections,selected_value_list=[""]):  #,*args,**kwargs):
    #    super(button_group,self).__init__(*args,**kwargs)
        self.name=name
        self.sub_name=sub_name
        self.document = pyglet.text.document.UnformattedDocument()
        self.x_start=x_start
        self.y_start=y_start
        self.x_len=x_len
        self.y_len=y_len
        self.layout=pyglet.text.layout.IncrementalTextLayout(self.document, x_len, y_len,multiline=True,wrap_lines=True, batch=batch)
        self.layout.x=x_start
        self.layout.width=x_len
        self.layout.y=y_start
        self.layout.height=y_len
        self.colour=colour
        self.inactive_colour=(40,50,60)
        self.pushed_colour=pushed_colour
        self.button_type=button_type
        self.title=title
        self.sub_title=sub_title
        self.active=active
        self.visible=visible
        self.movable=movable
        self.pushable=pushable
        self.floating=floating
        self.mouse_over=False
       # self.marked_pushed=pushed
        self.pushed=pushed
        self.toggle=toggle
        self.run_query_now=False
        self.mouse_over=mouse_over
        self.contains_list=contains_list
        self.unique_list=unique_list
        self.unique_list_start_point=unique_list_start_point
        self.allow_multiple_selections=allow_multiple_selections
        self.date_query=[]
        self.unique_list_display_length=30
        self.selected_value_list=[""]   #selected_value_list
        self.pageup_or_pagedown_jump=self.unique_list_display_length-1 #int(y_len/CHAR_HEIGHT)-1
      #  return self
      
        self.window=window
     
    
       # self.button_array=button_array
       # self.button_df=button_df.copy()
        
   

         
    def draw_button(self,k):
      #  print("draw button",k)

 #       for w in window_objects:
      #  for k,v in window_objects.items():
     #   print(k)
   #     window_objects["working"]["winobject"].add_text("draw button:"+str(k)+"\n")
        if self.window.window_objects[k]["object_type"]=="button":
            t=self.window.window_objects[k]["winobject"]
            if t.visible:
       #         print("t.name",t.name)
          #      window_objects["working"]["winobject"].add_text("drawing button:"+k+":"+t.name+"\n")
          #      window_objects["working"]["winobject"].add_text("button list:"+k+":"+str(t.unique_list)+"\n")
     #   for b in BUTTON_LIST:
     #       if b.visible:
                self._draw_button(t)
           # if b.visible & b.active:    
           #     position_list_in_active_window(x=b.x_start,y=b.y_start+40,input_list=b.selected_value_list)
         
        #return batch    
    
    
    
            
               
           
           
    def _draw_button(self,button): 
        batch = pyglet.graphics.Batch()
        if button.active:
            if button.contains_list:
                self.position_text_in_active_window(str(len(button.selected_value_list)-1),x=button.x_start,y=button.y_start+20)
            self.position_text_in_active_window(button.title,x=button.x_start,y=button.y_start-20)
            self.position_text_in_active_window(button.sub_title,x=button.x_start,y=button.y_start-35)        
            if button.button_type==1:
                self._draw_rect(button.x_start,button.y_start-60,button.x_len+10,button.y_len,colour=(255,0,0),batch=batch)
                
            if not button.pushed:
                self._draw_solid_rect(button.x_start,button.y_start,button.x_len,button.y_len,colour=button.colour,batch=batch)
            else:    
              #  batch=_draw_rect(button.x_start,button.x_len,button.y_start,button.y_len,colour=button.pushed_colour,batch=batch)
                self._draw_solid_rect(button.x_start,button.y_start,button.x_len,button.y_len,colour=button.pushed_colour,batch=batch)
     
        else:
           # batch=_draw_rect(button.x_start,button.x_len,button.y_start,button.y_len,colour=button.inactive_colour,batch=batch)       
            self._draw_solid_rect(button.x_start,button.y_start,button.x_len,button.y_len,colour=button.inactive_colour,batch=batch)       
        batch.draw()    
        
  
    
  


       
       
 
    

       
     
    def _draw_rect(self,x,y,x_len,y_len,colour,batch):
        final_colour=colour+colour
       # print("final colour=",final_colour)
        batch.add(2, pyglet.gl.GL_LINES, None,
                                  ('v2i', (x, y, x+x_len, y)),             
                                  ('c3B', final_colour)
        )
        batch.add(2, pyglet.gl.GL_LINES, None,
                                  ('v2i', (x+x_len, y, x+x_len, y+y_len)),             
                                  ('c3B', final_colour)
        )
        batch.add(2, pyglet.gl.GL_LINES, None,
                                  ('v2i', (x+x_len, y+y_len, x, y+y_len)),             
                                  ('c3B', final_colour)
        )
        batch.add(2, pyglet.gl.GL_LINES, None,
                                  ('v2i', (x, y+y_len, x, y)),             
                                  ('c3B', final_colour)
        )
        
        batch.draw()
         
        
     
        
        
     
    def _draw_solid_rect(self,x,y,x_len,y_len,colour,batch):
      #  final_colour=colour+colour
        
        
        rectangle = shapes.Rectangle(x, y, x_len, y_len, color=colour, batch=batch)
        rectangle.opacity = 128
        rectangle.rotation = 0
        
        batch.draw()
    #    # print("final colour=",final_colour)
     
    
    
    def position_text_in_active_window(self,text,*,x,y,color=(255,255,255,255)):
        batch = pyglet.graphics.Batch()
       # canvas={}
       # canvas[1] = pyglet.text.Label(text, x=x, y=y, batch=batch)
        pyglet.text.Label(text, x=x, y=y, batch=batch,color=color)
        batch.draw()
       # return batch
    

    
    #unique_list_start_point=1,
    def position_list_in_active_window(self,*,x,y,input_list,selection_list_input,color=(255,255,255,255)):
        batch = pyglet.graphics.Batch()
        for elem in input_list:
            if selection_list_input and (elem=="" or elem==" "):       
                elem="|__EMPTY__|"
         
            pyglet.text.Label(elem, x=x, y=y,batch=batch,color=color)
            y-=16
        batch.draw()
    
    
    
    def _display_text_in_active_window(self,text):
        batch = pyglet.graphics.Batch()
       # canvas={}
       # canvas[1] = pyglet.text.Label(text, x=x, y=y, batch=batch)
        pyglet.text.Label(text, x=5, y=window.get_size()[1]-12, batch=batch)
      #  window.clear()
        batch.draw()
     
    
       
            
    def _index_containing_substring(self,the_list, substring,start):
         if len(the_list)>0:
         #    print("ics=",the_list)
             for i, s in enumerate(the_list):
                 try:
                     if substring in s[0]:
                        return i
                 except:
                     print("Selection error",i,s,the_list)
             return start
         else:
             return start
         
     
    def _shortcut(self,char,x,y,window_objects):  
        for v in self.window.window_objects.values():  
           if v['object_type']=="button":  
               b=v["winobject"]    
               b.mouse_over=False
               if ((not b.floating) & b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                    b.mouse_over=True
    
     #        if ((not b.floating) & b.active & b.visible & b.mouse_over):
                    b.unique_list_start_point=self._index_containing_substring(b.unique_list,char,b.unique_list_start_point)
            #     self.move_and_draw_pointer(x,y)
      #           return True
         #return False   
      #    return b.selected_value_list[b.unique_list_start_point]
        
      
    def list_move(self,value,b,x,y): 
      #   t=window_objects[k]
       #  if tb["object_type"]=="button":
        #     b=tb["winobject"]
      #   for b in BUTTON_LIST:
         #    tb['mouse_over']=False
        #  if b['object_type']=="button":
             if ((not b.floating) & b.active & b.visible & b.mouse_over):    #(x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
            
    
             
         #    if ((not b.floating) & b.active & b.visible & b.mouse_over):
                 if (b.unique_list_start_point>=1) & (b.unique_list_start_point<=len(b.unique_list)-1):
                     b.unique_list_start_point+=value
                     if b.unique_list_start_point<1:
                         b.unique_list_start_point=1
                     if b.unique_list_start_point>=len(b.unique_list)-1:
                         b.unique_list_start_point=len(b.unique_list)-1
                #     self.move_and_draw_pointer(x,y)
        
     
     
 
      

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 
# class buttons(button):  
#     def __init__(self,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour,*args,**kwargs):
#          super().__init__(*args,**kwargs)  #group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour,*args,**kwargs)  # or super(B, self).__init_()
#   #  def create_button_group(self,*,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour):
#          self.group_name=group_name
#          self.group_x_start=group_x_start
#          self.group_x_len=group_x_len
#          self.group_y_start=group_y_start
#          self.group_y_len=group_y_len
#          self.group_border_colour=group_border_colour
#          self.group_border_on=group_border_on
#     #   pass
#         #  self.x=x
#        #  self.y=y


#     def create_buttons(self,x_start,y_start,filename,sales_df_dict,unique_dict):
#          bdf=pd.read_csv(filename,index_col=False,header=0)
#          i=0
#          windows_objects={}
#          for index, row in bdf.iterrows():
#              i+=1
#              colour=(int(row['colour1']),int(row['colour2']),int(row['colour3']))
#              pushed_colour=(int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
#            #  button=button_object(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),contains_list=bool(row['contains_list']),unique_list=[" "],allow_multiple_selections=True)    
#           #   BUTTON_LIST.append(button)
#           #   window_objects["button"+str(i)]={"object_type":"button","winobject":self.create_a_button(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,button_type=int(row['button_type']),pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),toggle=bool(row['toggle']),mouse_over=False,contains_list=bool(row['contains_list']),unique_list=[""],unique_list_start_point=1,allow_multiple_selections=True)}
#              self.create_a_button(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,button_type=int(row['button_type']),pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),toggle=bool(row['toggle']),mouse_over=False,contains_list=bool(row['contains_list']),unique_list=[""],unique_list_start_point=1,allow_multiple_selections=True)
            
#          return #window_objects
             
   
    
    
#    # def setup_buttons(self,window_objects,sales_df_dict,unique_dict):
#      #    x_start=20   
#      #    y_start=850
         
#      #    print("sales_df_dict=",sales_df_dict)
        
#          for fields,values in sales_df_dict.items():
#              if values==1:
#                  i+=1
#                  colour=(200,200,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
#                  pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
#                  window_objects["button"+str(i)]={"object_type":"button","winobject":new_button(name=str(fields),sub_name="AND",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title=str(fields),sub_title="AND",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields],unique_list_start_point=1,allow_multiple_selections=True)}
                 
#                 # button=button_object(name=str(fields),sub_name="AND",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title=str(fields),sub_title="AND",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
#               #   button.button_type=1   
#                 # BUTTON_LIST.append(button)
#                  i+=1
#                  x_start+=40
#                  colour=(200,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
#                  pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
#                  window_objects["button"+str(i)]={"object_type":"button","winobject":new_button(name=str(fields),sub_name="OR",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="",sub_title="OR",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields],unique_list_start_point=1,allow_multiple_selections=True)}
#                  #button=button_object(name=str(fields),sub_name="OR",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="OR",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
#                #  button.button_type=1   
#                  #BUTTON_LIST.append(button)
#                  i+=1
#                  x_start+=40
#                  colour=(0,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
#                  pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
#               #   button=button_object(name=str(fields),sub_name="NOT",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="NOT",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
#                  window_objects["button"+str(i)]={"object_type":"button","winobject":new_button(name=str(fields),sub_name="NOT",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="",sub_title="NOT",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields],unique_list_start_point=1,allow_multiple_selections=True)} 
#                #  button.button_type=1   
#                 # BUTTON_LIST.append(button)
                        
#                  x_start+=90
  
#              elif values==2:   # dfates
#                  if len(unique_dict['date_start'])>=731:
#                      uinitial_start_date_selection=[" ",unique_dict["date_start"][len(unique_dict['date_start'])-720]]
#                      ulist_start_point=len(unique_dict['date_start'])-720   #-356   #len(unique_dict['date_start'])-366
#                  else:
#                      uinitial_start_date_selection=[" ",unique_dict["date_start"][-1]]
#                      ulist_start_point=1
                    
            
#                   #      elif v==2:  # dates
#  #              unique_dict[k+"_start"]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
#  #               unique_dict[k+"_end"]=  unique_dict[k+"_start"][::-1]

#                  i+=4
#                  colour=(255,0,0)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
#                  pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
#                #  button=button_object(name=str(fields)+"_start",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="START",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields+"_start"],allow_multiple_selections=False)    
  
#                  window_objects["button"+str(i)]={"object_type":"button","winobject":new_button(name=str(fields)+"_start",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="START",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields+"_start"],unique_list_start_point=ulist_start_point,allow_multiple_selections=False,selected_value_list=uinitial_start_date_selection)} 
#                #  button.button_type=1
#                   #   BUTTON_LIST.append(button)
#                  x_start+=90
#                  i+=1
                 
                 
                 
             
#                  ulist_start_point=1
    
#                  uinitial_end_date_selection=[" ",unique_dict["date_end"][1]]  
    
    
#                  colour=(255,0,0)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
#                  pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
#                 # button=button_object(name=str(fields)+"_end",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="END",active=False,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields+"_end"],allow_multiple_selections=False)    
#                  window_objects["button"+str(i)]={"object_type":"button","winobject":new_button(name=str(fields)+"_end",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="END",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields+"_end"],unique_list_start_point=ulist_start_point,allow_multiple_selections=False,selected_value_list=uinitial_end_date_selection)}     
#                 # button.button_type=1   
#               #   BUTTON_LIST.append(button)
              
   
 
    
#                  x_start+=90
#             #     if i%2:   # st.display_number_of_records(sales_df,query_dict,query_df_list)
#             #        y_start+=10
#             #     else:   
#             #        y_start-=10

#              i+=6
                 
#      #    self._resize_buttons(window.x_max,window.y_max)
    
#          return window_objects
    
       
   
       
#     def _set_selections_except_date_start_inactive(self,window_objects):
#         for v in window_objects.values():  
#            if v['object_type']=="button":  
#                b=v["winobject"]
#                if not (b.name=="date_start" or b.name=="date_end" or b.name=="query_sales_df"):
#                    b.active=False
#     #    window.flip()
#         return
      


 
    
    # def _resize_buttons(self,x_max,y_max):
    #    # batch = pyglet.graphics.Batch()
    #     for button in BUTTON_LIST: 
    #         if button.x_start<0:
    #            button.x_start=0
    #         if button.x_start>x_max:
    #            button.x_start=x_max
    #         if button.y_start<0:
    #            button.y_start=0
    #         if button.y_start>y_max:
    #            button.y_start=y_max
        
        
    #         if button.x_len<0:
    #            button.x_len=0
    #         if button.x_len>x_max:
    #            button.x_len=x_max
    #         if button.y_len<0:
    #            button.y_len=0
    #         if button.y_len>y_max:
    #            button.y_len=y_max
        
        
    #         if button.x_start+button.x_len>x_max:
    #            button.x_len=x_max-button.x_start
            
    #         if button.y_start+button.y_len>y_max:
    #            button.y_len=y_max-button.y_start

              
       
   
    # def create_button_group(self,*,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour):
    #     self.group_name=group_name
    #     self.group_x_start=group_x_start
    #     self.group_x_len=group_x_len
    #     self.group_y_start=group_y_start
    #     self.group_y_len=group_y_len
    #     self.group_border_colour=group_border_colour
    #     self.group_border_on=group_border_on






#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class data_source(object):
    def __init__(self,window,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour,*args,**kwargs):
    #    super().__init__()  #name,sub_name,x_start,x_len,y_start,y_len,colour,pushed_colour,button_type,title,sub_title,active,visible,movable,pushable,floating,pushed,toggle,mouse_over,contains_list,unique_list,unique_list_start_point,allow_multiple_selections,selected_value_list=[""])  #,*args,**kwargs):group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour,*args,**kwargs)
        self.sales_df_dict={   # fields to display
            "date":2,  # two date fields, date_start, date_end
            
            "cat":0,
            "code":1,
            "costval":0,
            "doctype":0,
            "docentryno":0,
            "linenumber":0,
            "location":1,
            "product":1,
            "productgroup":1,
            "qty":0,
            "refer":0,
            "salesrep":1,
            "saleval":0,
            "territory":0,
            "glset":1,
            "specialpricecat":1,
            "sort":1,
            "period":0}
        
        # self.sales_df_types_dict={
        #     "cat":str,
        #     "code":str,
        #     "location":str,
        #     "product":str,
        #     "productgroup":str,
        #     "salesrep":str,
        #     "glset":str,
        #     "specialpricecat":int}
      #  self.sales_buttons=button_group
        self.qdf_size=0
        self.qdf_sales=0
        self.qdf_qty=0
        self.display=False
        self.query_dict_save={}
        
        self.query_save_list=[]
        
        self.display_x=30
        self.display_y=1000
        self.first_date_needed=True
        self.second_date_needed=False
        self.both_dates_entered=False
     #   self.xy=zip(np.array([0]),np.array([0]))
     #   self.saved_xy=zip(np.array([0]),np.array([0]))
        
        self.pat_cords=()
        self.pat_colors=()
        self.no_of_dots=0
        self.maxyval=0
        self.xval=0
        
        self.pareto_product_sprite=None
        self.yoy_dollars_sprite=None
        self.pareto_customer_sprite=None
        self.mat_sprite=None
        self.monthly_summary_sprite=None
        
     

  #  def create_button_group(self,*,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour):
        self.window=window
   #     print("window.window objects=",self.window.window_objects)
      
        self.group_name=group_name
        self.group_x_start=group_x_start
        self.group_x_len=group_x_len
        self.group_y_start=group_y_start
        self.group_y_len=group_y_len
        self.group_border_colour=group_border_colour
        self.group_border_on=group_border_on
    #   pass
        #  self.x=x
       #  self.y=y


    def create_buttons(self,x_start,y_start,filename,sales_df_dict,unique_dict):
         bdf=pd.read_csv(filename,index_col=False,header=0)
         i=0
         windows_objects={}
         for index, row in bdf.iterrows():
             i+=1
             colour=(int(row['colour1']),int(row['colour2']),int(row['colour3']))
             pushed_colour=(int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
           #  button=button_object(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),contains_list=bool(row['contains_list']),unique_list=[" "],allow_multiple_selections=True)    
          #   BUTTON_LIST.append(button)
         #    button=button(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,button_type=int(row['button_type']),pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),toggle=bool(row['toggle']),mouse_over=False,contains_list=bool(row['contains_list']),unique_list=[""],unique_list_start_point=1,allow_multiple_selections=True)
             self.window.window_objects["button"+str(i)]={"object_type":"button","winobject":button(window=self.window,name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,button_type=int(row['button_type']),pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),toggle=bool(row['toggle']),mouse_over=False,contains_list=bool(row['contains_list']),unique_list=[""],unique_list_start_point=1,allow_multiple_selections=True)}
          #   self.create_a_button(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,button_type=int(row['button_type']),pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),toggle=bool(row['toggle']),mouse_over=False,contains_list=bool(row['contains_list']),unique_list=[""],unique_list_start_point=1,allow_multiple_selections=True)
          #  
        # return #window_objects
             
   
    
    
   # def setup_buttons(self,window_objects,sales_df_dict,unique_dict):
     #    x_start=20   
     #    y_start=850
         
         print("sales_df_dict=",self.sales_df_dict)
        
         for fields,values in self.sales_df_dict.items():
             if values==1:
                 i+=1
                 colour=(200,200,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 self.window.window_objects["button"+str(i)]={"object_type":"button","winobject":button(window=self.window,name=str(fields),sub_name="AND",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title=str(fields),sub_title="AND",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields],unique_list_start_point=1,allow_multiple_selections=True)}
                 
                # button=button_object(name=str(fields),sub_name="AND",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title=str(fields),sub_title="AND",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
              #   button.button_type=1   
                # BUTTON_LIST.append(button)
                 i+=1
                 x_start+=40
                 colour=(200,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 self.window.window_objects["button"+str(i)]={"object_type":"button","winobject":button(window=self.window,name=str(fields),sub_name="OR",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="",sub_title="OR",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields],unique_list_start_point=1,allow_multiple_selections=True)}
                 #button=button_object(name=str(fields),sub_name="OR",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="OR",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
               #  button.button_type=1   
                 #BUTTON_LIST.append(button)
                 i+=1
                 x_start+=40
                 colour=(0,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
              #   button=button_object(name=str(fields),sub_name="NOT",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="NOT",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
                 self.window.window_objects["button"+str(i)]={"object_type":"button","winobject":button(window=self.window,name=str(fields),sub_name="NOT",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="",sub_title="NOT",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields],unique_list_start_point=1,allow_multiple_selections=True)} 
               #  button.button_type=1   
                # BUTTON_LIST.append(button)
                        
                 x_start+=90
  
             elif values==2:   # dfates
                 if len(unique_dict['date_start'])>=731:
                     uinitial_start_date_selection=[" ",unique_dict["date_start"][len(unique_dict['date_start'])-720]]
                     ulist_start_point=len(unique_dict['date_start'])-720   #-356   #len(unique_dict['date_start'])-366
                 else:
                     uinitial_start_date_selection=[" ",unique_dict["date_start"][-1]]
                     ulist_start_point=1
                    
            
                  #      elif v==2:  # dates
 #              unique_dict[k+"_start"]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
 #               unique_dict[k+"_end"]=  unique_dict[k+"_start"][::-1]

                 i+=4
                 colour=(255,0,0)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
               #  button=button_object(name=str(fields)+"_start",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="START",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields+"_start"],allow_multiple_selections=False)    
  
                 self.window.window_objects["button"+str(i)]={"object_type":"button","winobject":button(window=self.window,name=str(fields)+"_start",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="START",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields+"_start"],unique_list_start_point=ulist_start_point,allow_multiple_selections=False,selected_value_list=uinitial_start_date_selection)} 
               #  button.button_type=1
                  #   BUTTON_LIST.append(button)
                 x_start+=90
                 i+=1
                 
                 
                 
             
                 ulist_start_point=1
    
                 uinitial_end_date_selection=[" ",unique_dict["date_end"][1]]  
    
    
                 colour=(255,0,0)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                # button=button_object(name=str(fields)+"_end",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="END",active=False,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields+"_end"],allow_multiple_selections=False)    
                 self.window.window_objects["button"+str(i)]={"object_type":"button","winobject":button(window=self.window,name=str(fields)+"_end",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,button_type=1,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="END",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,toggle=True,mouse_over=False,contains_list=True,unique_list=unique_dict[fields+"_end"],unique_list_start_point=ulist_start_point,allow_multiple_selections=False,selected_value_list=uinitial_end_date_selection)}     
                # button.button_type=1   
              #   BUTTON_LIST.append(button)
              
   
 
    
                 x_start+=90
            #     if i%2:   # st.display_number_of_records(sales_df,query_dict,query_df_list)
            #        y_start+=10
            #     else:   
            #        y_start-=10

             i+=6
                 
     #    self._resize_buttons(window.x_max,window.y_max)
    
   #      return window_objects
    
       
    
       
#     def _draw_button_group_border(self):      
#         pyglet.gl.glLineWidth(1)
#      #   print("1",data_source)
#     #    print("2",sales_buttons.data_source)
#        # print("3",sales)
# #        print("4",sales_buttons)
#       #  print("5",self.group_y_start)
#         outline = batch.add(4, pyglet.gl.GL_LINE_LOOP, None, ('v2f', (self.group_x_start, self.group_y_start, self.group_x_start+self.group_x_len, self.group_y_start, self.group_x_start+self.group_x_len, self.group_y_start+self.group_y_len, self.group_x_start, self.group_y_start+self.group_y_len)), ('c4B', self.group_border_colour*4))    #('c4B', (255, 0, 0, 0)*4))
        

    
    
   
       
    def _set_selections_except_date_start_inactive(self,window_objects):
        for v in self.window.window_objects.values():  
           if v['object_type']=="button":  
               b=v["winobject"]
               if not (b.name=="date_start" or b.name=="date_end" or b.name=="query_sales_df"):
                   b.active=False
    #    window.flip()
        return
      


 
    
    # def _resize_buttons(self,x_max,y_max):
    #    # batch = pyglet.graphics.Batch()
    #     for button in BUTTON_LIST: 
    #         if button.x_start<0:
    #            button.x_start=0
    #         if button.x_start>x_max:
    #            button.x_start=x_max
    #         if button.y_start<0:
    #            button.y_start=0
    #         if button.y_start>y_max:
    #            button.y_start=y_max
        
        
    #         if button.x_len<0:
    #            button.x_len=0
    #         if button.x_len>x_max:
    #            button.x_len=x_max
    #         if button.y_len<0:
    #            button.y_len=0
    #         if button.y_len>y_max:
    #            button.y_len=y_max
        
        
    #         if button.x_start+button.x_len>x_max:
    #            button.x_len=x_max-button.x_start
            
    #         if button.y_start+button.y_len>y_max:
    #            button.y_len=y_max-button.y_start

              
       
   
    # def create_button_group(self,*,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour):
    #     self.group_name=group_name
    #     self.group_x_start=group_x_start
    #     self.group_x_len=group_x_len
    #     self.group_y_start=group_y_start
    #     self.group_y_len=group_y_len
    #     self.group_border_colour=group_border_colour
    #     self.group_border_on=group_border_on








        
        
    
    def load_pickle(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            print("\nLoading "+str(save_dir)+str(savefile)+"...")
            window_objects["working"]["winobject"].add_text("\nLoading "+str(save_dir)+str(savefile)+"...")
            return self.preprocess(pd.read_pickle(save_dir+savefile))
        else:
      #      print("load sales_df error.")
            window_objects["working"]["winobject"].add_text("load sales_df error.\n")
            return
        
    
    def preprocess(self,df):    #, rename_dict):
     #   print("preprocess_sc sales save df=",df,rename_dict)
        df=df[(df['code']!="OFFINV")]   
        df=df[(df['product']!="OFFINV")]   
        
        df=df[(df['product']!=0)]   

     #   print("\nPreprocess data to convert GSV to NSV exclude products="+str(dd2.bbtv_dict['sales']["GSV_prod_codes_to_exclude"]))
        self.window.window_objects["working"]["winobject"].add_text("\nPreprocess data to convert GSV to NSV exclude products="+str(dd2.bbtv_dict['sales']["GSV_prod_codes_to_exclude"])+"\n")
        return df[~df['product'].isin(dd2.bbtv_dict['sales']["GSV_prod_codes_to_exclude"])]  # .copy() 
 


    def find_uniques(self,sales_df):
       # position_text_in_active_window("Sales Transactions size="+str(sales_df.shape[0])+" rows",x=0,y=0)
     #   print("Indexing selection columns...")
#        window_objects["working"]["winobject"].add_text("Indexing selection columns...\n")
        unique_dict={}
        for k,v in self.sales_df_dict.items():
            if v==1:
                if k=="specialpricecat":
                    unique_dict[k]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
                 #   unique_dict[k]=[int(i) for i in unique_dict[k]]
                else:    
                    unique_dict[k]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
            elif v==2:  # dates
                unique_dict[k+"_start"]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
                unique_dict[k+"_end"]=  unique_dict[k+"_start"][::-1]

     #   print("find uniqeus=",unique_dict)
    #    print("Ready.\n")
 #       window_objects["working"]["winobject"].add_text("Ready.\n")
 
        return unique_dict



    def display_number_of_records(self,sales_df,query_dict,query_df_list,x,y,plot_output_dir):
        
        print_flag=False
     #   pyglet.TextLayout.color=(255,255,255,255) 
      #  ir.position_text_in_active_window("To Select:Mouse over the fields rect, then PageUp/Pagedown/up/down arrows/mouse scroll wheel or type first letter. Press F1-add a selection, F2-remove the selected val. Esc to exit.",x=x,y=y,color=(255,100,200,255))
      #  ir.position_text_in_active_window("Total Sales Transactions size="+'{:,.0f}'.format(sales_df.shape[0])+" rows",x=x,y=y-15)
      #  self.total_sales=sales_df['salesval'].sum()
      #  self.total_qty=int(sales_df['qty'].sum())
      #  ir.position_text_in_active_window("Total Sales Transactions NSV value "+'${:,.2f}'.format(self.total_sales),x=x,y=y-30)
      #  ir.position_text_in_active_window("Total Sales Unit Qty="+'{:,.0f}'.format(self.total_qty),x=x,y=y-45)
        
      #  if len(query_df_list)>0:
        #    print("query df list=\n",query_df_list)   
        if len(query_df_list)==1:
            qdf=query_df_list[0]
            if self.qdf_size!=qdf.shape[0]: 
        
                self.qdf_size=qdf.shape[0]
           #     print("Query complete",self.qdf_size,"records") 
                self.window.window_objects["results"]["winobject"].add_text("Query complete. "+str(self.qdf_size)+" records\n")
              #  window_objects["query_queue"]["winobject"].add_text("Query complete. "+str(self.qdf_size)+" records\n")
                report_query_info(qdf)
                if self.qdf_size>0:
            #        print("plotting.",end="\r",flush=True)
                    self.window.window_objects["working"]["winobject"].add_text("Plotting.\n")
                    
                    self.plot_mat(qdf,str(query_dict),plot_output_dir)
                                    
                    self.qdf_sales=qdf['salesval'].sum()
                    self.qdf_qty=int(qdf['qty'].sum())
                    self.display=True
    
                    self.mat_sprite = pyglet.sprite.Sprite(pyglet.image.load(plot_output_dir+'ir_mat_chart.png'))
                    self.mat_sprite.x=1240
                    self.mat_sprite.y=900
                    self.mat_sprite.scale_y=0.5 #0.36
                    self.mat_sprite.scale_x=0.55 #0.34
                    self.mat_sprite.opacity=200
                    
                    self.plot_pareto_customer(qdf,str(query_dict),plot_output_dir)
                    self.pareto_customer_sprite = pyglet.sprite.Sprite(pyglet.image.load(plot_output_dir+'ir_pareto_customer_chart.png'))
                    self.pareto_customer_sprite.x=60
                    self.pareto_customer_sprite.y=300
                    self.pareto_customer_sprite.scale_y=0.5 #0.36
                    self.pareto_customer_sprite.scale_x=0.6 #0.39
                    self.pareto_customer_sprite.opacity=200
                    
                    self.plot_yoy_dollars(qdf,str(query_dict),plot_output_dir)
                    self.yoy_dollars_sprite = pyglet.sprite.Sprite(pyglet.image.load(plot_output_dir+'ir_yoy_dollars_chart.png'))
                    self.yoy_dollars_sprite.x=1240
                    self.yoy_dollars_sprite.y=300
                    self.yoy_dollars_sprite.scale_y=0.5 #0.35
                    self.yoy_dollars_sprite.scale_x=0.55 #0.41
                    self.yoy_dollars_sprite.opacity=200
                    
                    self.plot_pareto_product(qdf,str(query_dict),plot_output_dir)
                    self.pareto_product_sprite = pyglet.sprite.Sprite(pyglet.image.load(plot_output_dir+'ir_pareto_product_chart.png'))
                    self.pareto_product_sprite.x=60
                    self.pareto_product_sprite.y=900
                    self.pareto_product_sprite.scale_y=0.5 #0.36
                    self.pareto_product_sprite.scale_x=0.6 #0.39
                    self.pareto_product_sprite.opacity=200
                   
                    
                   
                    # make a UUID using an MD5 hash of a namespace UUID and a name
                    unique_name= uuid.uuid1()    #uuid.NAMESPACE_DNS, 'python.org')
                 #   print("uuid unique suffix name",unique_name)
                #    os.copy()
                    copyfile(plot_output_dir+'ir_mat_chart_normal.png', plot_output_dir+'ir_mat_chart_'+str(unique_name)+".png")
                    copyfile(plot_output_dir+'ir_pareto_customer_chart_normal.png', plot_output_dir+'ir_pareto_customer_chart_'+str(unique_name)+".png")                    
                    copyfile(plot_output_dir+'ir_pareto_product_chart_normal.png', plot_output_dir+'ir_pareto_product_chart_'+str(unique_name)+".png")
                    copyfile(plot_output_dir+'ir_yoy_dollars_chart_normal.png', plot_output_dir+'ir_yoy_dollars_chart_'+str(unique_name)+".png")
                    
                    
                    self.window.window_objects["working"]["winobject"].add_text("Finished plotting.....\n")
                    
                   
              
                  #  result_sprites = []
            
           #         result_sprites.append(pyglet.sprite.Sprite(self.pareto_product_sprite, batch=batch))
           #         result_sprites.append(pyglet.sprite.Sprite(self.pareto_customer_sprite, 210, 10, batch=batch))
           #         result_sprites.append(pyglet.sprite.Sprite(self.mat_sprite, 410, 10, batch=batch))
           #         result_sprites.append(pyglet.sprite.Sprite(self.yoy_dollars_sprite, 610, 10, batch=batch))
               
                    
                    
                    
                #    self.pat_cords,self.pat_colors,self.no_of_dots,self.xval,self.maxyval=self.plot_salesval_activity_timeline(qdf)
                    
                 #   s = pyglet.sprite.Sprite(self.water_tile, 100, 100, batch=sa_batch)
                    # self.sales_activity_sprite = pyglet.sprite.Sprite(pyglet.image.load(plot_output_dir+'sales_activity_chart.png'))
                    # self.sales_activity_sprite.x=10
                    # self.sales_activity_sprite.y=10
                    # self.sales_activity_sprite.scale_y=0.43
                    # self.sales_activity_sprite.scale_x=0.39
                    # self.sales_activity_sprite.opacity=200
    
                 #   print("plotting....... complete. Waiting.")
                    self.window.window_objects["working"]["winobject"].add_text("plotting....... complete. Waiting.\n")
            #    self.quick_mat(query_df_list,x,y)  
        #    else:
        #        self.display=False
        if self.display:
            if query_dict:
                self.query_dict_save=query_dict
              #  if len(str(self.query_dict_save))>200 and not print_flag:
              #      print("query too long to display in window=",str(self.query_dict_save))
              #      print_flag=True
            query_name=str(self.query_dict_save)      
#            position_text_in_active_window("Query="+str(self.query_dict_save),x=x,y=y-65)
       #     ir.position_text_in_active_window("Query="+query_name[:170],x=x,y=y-65)
       #     ir.position_text_in_active_window(query_name[170:350],x=x,y=y-80)
       #     ir.position_text_in_active_window(query_name[350:],x=x,y=y-95)

         #   self.pareto_product_sprite.draw()
            
            start_date=pd.to_datetime('today')
            end_date=pd.to_datetime('today')
       #     for c in BUTTON_LIST:
            for v in self.window.window_objects.values():  
               if v['object_type']=="button":  
                  c=v["winobject"]    
            
                  if (c.name=="date_start") & (len(c.selected_value_list)>1):
                       start_date=pd.to_datetime(c.selected_value_list[-1])
               #     print(start_date)
                  if (c.name=="date_end") & (len(c.selected_value_list)>1):
                       end_date=pd.to_datetime(c.selected_value_list[-1])
                #    print(end_date)
            if isinstance(start_date,pd.Timestamp) & isinstance(end_date,pd.Timestamp):  
                query_span=(end_date-start_date).days
                        
                self.window.window_objects["results"]["winobject"].add_text("Query day length="+str(query_span)+" days")
        #     ir.position_text_in_active_window("Queried Sales Transactions size="+'{:,.0f}'.format(self.qdf_size)+" rows",x=x,y=y-130)
        #     ir.position_text_in_active_window("Query Transactions NSV value "+'${:,.2f}'.format(self.qdf_sales),x=x,y=y-145)
        #     ir.position_text_in_active_window("Query Unit Qty="+'{:,.0f}'.format(self.qdf_qty),x=x,y=y-160)
        # #    self.plot_salesval_activity_timeline(qdf)
        #     ir.position_text_in_active_window("$"+str(self.maxyval)+" Max/day",x=self.xval,y=550)
        #     batch = pyglet.graphics.Batch()
   
        #     batch.add(self.no_of_dots, gl.GL_POINTS, None, ('v2i', self.pat_cords), ('c3B', self.pat_colors))
            
        #     batch.draw()
        #  window.activate()
  
        
        
         #   self.sprite.x=x
          #  self.sprite.y=y-500
       #     batch = pyglet.graphics.Batch()
         #   self.mat_sprite.draw()
         #   self.yoy_dollars_sprite.draw()
         #   self.sales_activity_sprite.draw()
   #         for c in BUTTON_LIST:
       #     self.pareto_product_sprite.draw() 
       #     self.yoy_dollars_sprite.draw()
            
            
           
            
            
            
            
            
            
            # for v in window_objects.values():  
            #    if v['object_type']=="button":  
            #       c=v["winobject"]    
      
                
                
                
            #       if c.name=="pareto_type":
            #            if c.pushed:   
            #                self.pareto_product_sprite.draw()
            #            else:
            #                self.pareto_customer_sprite.draw()
            #       if c.name=="dollar_report_type":
            #            if c.pushed:   
            #                self.yoy_dollars_sprite.draw()
            #            else:
            #                self.mat_sprite.draw()

                        
         #       if c.name=="pareto_mp4":
         #           if c.pushed:
         #               print("pareto mp4")
        #    self.pareto_product_sprite.draw()
        #    self.pareto_customer_sprite.draw()
  
       #     batch.draw()    
          #  self.quick_mat(query_df_list,x,y)
            #self.xy=self.quick_mat(query_df_list,x,y)  
        #    self.quick_mat(query_df_list,x,y)  
          #  self._graph_mat(self.saved_xy)   #x_vals,scaled_y_vals)
            #self.quick_mat(self.xy)                     
        return
   
    


    def create_query(self):
       # query sales_df to produce queries_slaes_df
     #   window_objects["working"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")
        
        #    window_objects["query_queue"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")
        #    window_objects["results"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")

        #print("create query doct in=",query_dict)
        stretched_query=[]
        query_list=[]
       # for b in BUTTON_LIST:
        for v in self.window.window_objects.values():  
            if v['object_type']=="button":  
                b=v["winobject"]    
  
       #     print("selvl=",b.selected_value_list)
                qlist=b.selected_value_list[1:]
 
                if len(qlist)>0:
                #    b.occupied=True
                 #   print("create query",b.name)
                    if b.name=="specialpricecat":
 #                       stretched_query=[b.sub_name]+[(b.name,int(i)) if isinstance(i, numbers.Number) else (b.name,i) for i in qlist]
                        stretched_query=[b.sub_name]+[(b.name,float(i)) if type(i) == int or float else (b.name,i) for i in qlist]
 
                   #     print("sq=",stretched_query) 
                        query_list.append(stretched_query)
                   #     print("spc streched query=",stretched_query) 
                      #  query_list.append(stretched_query)
                
                    elif b.name=="date_start":   # and self.first_date_needed:   # and not self.second_date_needed:   
                    #    print(" date start qlist=",qlist)
                        for i in qlist:
                      #      self.date_query=[(b.name,pd.to_datetime(i),pd.to_datetime(i))]
                            self.date_query=[['date',pd.to_datetime(i),pd.to_datetime(i)]]
                            
                       #     print(" start selected value=",b.selected_value_list)
                      #      if len(b.selected_value_list)>1:
                      #          self.first_date_needed=False
                            self.first_date_needed=False       
                            self.second_date_needed=True
                            self.both_dates_entered=False
                            
                           # for c in BUTTON_LIST:
                            for v in window_objects.values():  
                                if v['object_type']=="button":  
                                    c=v["winobject"]    
     
                                    if c.name=='date_end':
                                       c.active=True
                                       break
                        #    BUTTON_LIST['date_end'].active=True
                     #       print("first date date_query==",self.date_query)
                          #  s=self.grouper(2,t)
                    #        print("dtae start=",self.date_query)
                          #  stretched_query=[b.sub_name]+t
                        
                      
                    elif b.name=="date_end" and (not self.first_date_needed):   # and self.second_date_needed:   
                     #   print("date end qlist=",qlist)
                        b.active=True
                        for i in qlist:
                      #      self.date_query=[(b.name,pd.to_datetime(i),pd.to_datetime(i))]
                        #    print("before date_query=",self.date_query)
                        #    print("bdq02=",b.date_query[0][2])
                        #    print("i=",i)
                            self.date_query[0][2]=pd.to_datetime(i)
                                
                      #      print("after change self.date_query=",self.date_query)
                            b.date_query=[tuple(self.date_query[0])]
                       #     print("after2 b.date_query=",b.date_query)
                       #     print(" finished selected value=",b.selected_value_list)
                            self.first_date_needed=False 
                        #    if len(b.selected_value_list)>1:
                        #        self.second_date_needed=False
                            self.second_date_needed=False
                            self.both_dates_entered=True
                        #    print("second date date_query==",b.date_query)
                          #  s=self.grouper(2,t)
                           # print("s=",s)
                            stretched_query=[b.sub_name]+b.date_query
                            query_list.append(stretched_query)
                            self.date_query=[]
                     #   print("date end sq=",stretched_query)
                       
                        
                    else:                       
                        stretched_query=[b.sub_name]+[(b.name,i) for i in qlist]

                        query_list.append(stretched_query)
                        self.both_dates_entered=not (self.first_date_needed or self.second_date_needed)

                     #   print("streched query=",stretched_query) 
                        
                        
                        
        if self.both_dates_entered:

            self.window.window_objects["working"]["winobject"].add_text("\n\n.....QUERYING...Please wait.\n\n")
            self.window.window_objects["working"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n") 
            self.window.window_objects["query_queue"]["winobject"].add_text("working on query "+str(query_list)+"\n")
            self.window.window_objects["results"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")       
            self._unset_selections_except_date_start_inactive()
            self.window.flip()
         #   window.clear()
            self.window.render()
            self.window.flip()
       #     time.sleep(1)
          #  window.flip()
 
  #          window.flip()
           # pyglet.clock.schedule_once(self._undraw_querying_sign, 6.0)
            
        #    pyglet.clock.schedule_interval(window._draw_querying_sign, 0.1)
  #          window.querying_sign_on=True
       #    stbg.draw_button_group_border()
        #    window_objects["working"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")
         #   window_objects["query_queue"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")
         #   window_objects["results"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")
            
         #   window.flip()
        #    print("Working on query:",query_list)
            query_dict={"query":query_list} 
            query_df_list=self.queries(query_dict)
         #   pyglet.clock.unschedule(window._draw_querying_sign)
            window.querying_sign_on=False
          #  stbg.draw_button_group_border()
            if query_df_list is not None:
                self.query_save_list=query_df_list
            return query_dict,query_df_list
        else:
            return {},[]
                    
   

    def _unset_selections_except_date_start_inactive(self):
        for v in self.window.window_objects.values():  
           if v['object_type']=="button":  
               b=v["winobject"]
               if not b.name=="date_start":
                   b.active=True
   #     window.flip()
        return
  

       
        
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
                        try:
                           new_df=new_df[(new_df[field]==q[1])].copy() 
                        except:
                           print("AND query error",q)
                           self.window.window_objects["working"]["winobject"].add_text("AND query error. "+str(q))
  
                  #      print("AND query=",field,"==",q[1],"\nnew_df=",new_df.shape) 
                  #      print("new new_df=\n",new_df)    
                elif oper=="OR":
                    new_df_list=[]
                    for q in query_list:   
                        try:
                           new_df_list.append(new_df[(new_df[q[0]]==q[1])].copy())                        
                        except:
                           print("OR query error",q)
                           self.window.window_objects["working"]["winobject"].add_text("OR query error. "+str(q))
 
  
                     #   print("OR query=",q,"|",new_df_list[-1].shape)
                    new_df=new_df_list[0]    
                    for i in range(1,len(query_list)):    
                        new_df=pd.concat((new_df,new_df_list[i]),axis=0)   
                  #  print("before drop",new_df.shape)    
                    new_df.drop_duplicates(keep="first",inplace=True)   
                  #  print("after drop",new_df.shape)
                elif oper=="NOT":
                    for q in query_list:    
                        try:
                           new_df=new_df[(new_df[q[0]]!=q[1])].copy() 
                        except:
                           print("NOT query error",q)
                           self.window.window_objects["working"]["winobject"].add_text("NOT query error. "+str(q))
 
  
                           
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
                        try:
                           end=q[2]
                        except IndexError:
                           end=q[1] 
                        try:   
                           new_df=new_df[(pd.to_datetime(new_df[q[0]])>=pd.to_datetime(q[1])) & (pd.to_datetime(new_df[q[0]])<=pd.to_datetime(q[2]))].copy() 
                        except:
                            self.window.window_objects["working"]["winobject"].add_text("between dates query error. "+str(q))
                            print("between dates query error",q)
                            
  
                        
 #       print("Beeterm AND query=",q,"&",new_df.shape) 
                   # else:
                   #     print("Error in between statement")
                elif oper=="B":  # btween numbers or strings
                  #  if (len(query_list[0])==3):
                    for q in query_list:
                    #    print("between ql=",q[1],q[2])
                    
                        start=q[1]
                        try:
                            end=q[2]
                        except IndexError:
                            end=q[1]
                        try:    
                            new_df=new_df[(new_df[q[0]]>=q[1]) & (new_df[q[0]]<=q[2])].copy() 
                        except:
                            print("between query error",q)
                            self.window.window_objects["working"]["winobject"].add_text("between query error. "+str(q))
 
                            
                     #       print("Beeterm AND query=",q,"&",new_df.shape) 
                   # else:
                   #     print("Error in between statement")
     
                else:
                    print("operator not found\n")
                    self.window.window_objects["working"]["winobject"].add_text("operator not found.")
 
                        
                return new_df.copy()
                      
           else:
                print("invalid operator")
                self.window.window_objects["working"]["winobject"].add_text("invalid operator.")
                return pd.DataFrame([])
    
  
      
    def _build_a_query(self,query_spec):
     #   print("build an entry query_name",query_name)
      #   print("filesave name=",query_name[0])
        #queries=query_name[1]
      #  query_name=qd.queries[q]
        new_df=sales_df.copy()
      #  q_df=pd.DataFrame([])
      #  new_df=query_df.copy()
        for qn in query_spec:  
      #      print("build a query dict qn=",qn)
            q_df=self._query_df(new_df,qn)
            new_df=q_df.sort_index(ascending=False,axis=0).copy()
       #     print("new+df=\n",new_df)
        new_df.drop_duplicates(keep="first",inplace=True)    
       # q_df=smooth(q_df)
     #   self.save(q_df,dd2.bbtv_dict['sales']['save_dir'],query_name[0])   
        return new_df
    
           
    
    def queries(self,query_dict):
      #  self.query=sales_query_class()
      
     #   query_df=qdf.copy()
     #   dd2.bbtv_dict['sales']['query_df']=query_df.copy()
    

        if len(query_dict)>0:
         #   df=df.rename(columns=qd.rename_columns_dict)  
          #  query_handles=[]
            query_df_list=[]
            for v in query_dict.values():
                query_df_list.append(self._build_a_query(v))   #st.save_query(q_df,query_name,root=False)   
         #   query_filenames=[q[:250] for q in query_handles[0]]  # if len(q)>249]
         #   print("build a query dict query filenames",query_filenames)
            return query_df_list   #{k: v for k, v in zip(query_dict.keys(),query_df_list[0])}     #,{k: v for k, v in zip(qd.queries.keys(),query_filenames)}
        else:
            print("query dict empty")
            self.window.window_objects["working"]["winobject"].add_text("query dict empty")
            return []
    
   
    
    def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
    


    def plot_mat(self,mat_df,query_name,plot_output_dir):
      new_df=mat_df.groupby(mat_df.index,sort=True).sum()
   #   print("today=",pd.to_datetime("today").dt.days)
      #todays_date = dt.date.today()
    #  start_mat_date=(pd.to_datetime('today')+timedelta(-731)).strftime("%Y-%m-%d") 
    #  end_mat_date=new_df.index[-1].strftime("%Y-%m-%d") 
      
      end_mat_date=(new_df.index[-1]).strftime("%Y-%m-%d")     #pd.to_datetime('today').strftime("%Y-%m-%d") 
      neg365_mat_date=(new_df.index[-1]+timedelta(-365)).strftime("%Y-%m-%d") 
     # neg731_mat_date=(new_df.index[-1]+timedelta(-731)).strftime("%Y-%m-%d") 

     # print(end_mat_date,neg365_mat_date,neg731_mat_date)
   #   window_objects["results"]["winobject"].add_text(str(end_mat_date)+"-"+str(neg365_mat_date)+"-"+str(neg731_mat_date))
      new_df['mat']=new_df['salesval'].rolling(365,axis=0).sum().round(0)
     # print(new_df.loc[neg365_mat_date,'mat'],new_df.loc[end_mat_date,'mat'])
      annual_growth_rate=round((new_df.loc[end_mat_date,'mat']-new_df.loc[neg365_mat_date,'mat'])/new_df.loc[end_mat_date,'mat']*100,1)
     # print("MAT Annual growth rate=",annual_growth_rate)
      self.window.window_objects["results"]["winobject"].add_text("MAT Annual growth rate="+str(annual_growth_rate)+"%\n")
    #  annual_growth_rate=1
      
 
      styles1 = ['r-']
    # styles1 = ['bs-','ro:','y^-']
      linewidths = 2  # [2, 1, 4]
             
      fig, ax = pyplot.subplots()   #(figsize=(6,4))
      fig.patch.set_facecolor('None')
      fig.autofmt_xdate()
      ax=new_df.iloc[365:][['mat']].plot(use_index=True,grid=True,fontsize=6,style=styles1, lw=linewidths,legend=False)
      ax.patch.set_facecolor('None')
      ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places
    #  ax.xaxis.set_major_formatter(StrMethodFormatter('{yyyy/mmm}')) # 2 decimal places

      ax.set_title(query_name[:105]+"\n"+query_name[105:205]+"\n"+query_name[205:300]+" $ sales MAT. Annual growth rate="+str(annual_growth_rate)+" %.",fontsize= 6,color='white')
    #  ax.legend(title="",fontsize=6)
      ax.set_xlabel("",fontsize=6)
      ax.set_ylabel("",fontsize=6)
     # ax.yaxis.set_major_formatter('${x:1.0f}')
      ax.yaxis.set_tick_params(which='major', labelcolor='white',
                   labelleft=True, labelright=False)
      ax.xaxis.set_tick_params(which='major', labelcolor='white')
    #  ax.grid(color="w",linestyle="-")
   #   plt.tight_layout()
      #plt.show()
      self._save_fig("ir_mat_chart",plot_output_dir,transparent=True,facecolor=fig.get_facecolor())  #self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
      self._save_fig("ir_mat_chart_normal",plot_output_dir,transparent=False,facecolor='black') 
      plt.close()
      
      return   
 
      


    def plot_pareto_customer(self,par_df,query_name,plot_output_dir):
        top=60
        new_df=par_df.groupby(['code'],sort=False).sum()
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
            fig.patch.set_facecolor('None')
            fig.autofmt_xdate()
          #  ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places

#                ax.yaxis.set_major_formatter('${x:1.0f}')
          #  ax.yaxis.set_tick_params(which='major', labelcolor='green',
          #           labelleft=True, labelright=False)

         #   ax.ticklabel_format(style='plain') 
         #   ax.yaxis.set_major_formatter(ScalarFormatter())
      
            #ax.ticklabel_format(style='plain') 
      #      ax.axis([1, 10000, 1, 100000])
            
            ax=new_df.plot.bar(y='salesval',ylabel="",fontsize=7,grid=False,legend=False)
            ax.patch.set_facecolor('None')
        #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
       #     axis.set_major_formatter(ScalarFormatter())
         #   ax.ticklabel_format(style='plain')
            ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places

            ax.yaxis.set_tick_params(which='major', labelcolor='white',labelleft=True, labelright=False)

        #    ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" customer $ ranking total dollars "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
            ax.set_title(query_name[:105]+"\n"+query_name[105:205]+"\n"+query_name[205:300]+" $ sales customer pareto",fontsize= 6,color='white')       
         
            ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=False,style=["w-"],secondary_y=True,legend=False)
            ax.xaxis.set_tick_params(labelcolor='white')
            ax2.xaxis.set_tick_params(labelcolor='white')
            ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
            ax2.yaxis.set_tick_params(labelcolor='white')
            ax3 = ax.twiny() 
            ax3.xaxis.set_tick_params(labelcolor='white')
            ax3.yaxis.set_tick_params(labelcolor='white')
            ax4=new_df[['ccount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
            if df_len<=1:
                df_len=2
     
            
            ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
            ax4.xaxis.set_tick_params(labelcolor='white')
            ax4.yaxis.set_tick_params(labelcolor='white')
            
            self._save_fig("ir_pareto_customer_chart",plot_output_dir,transparent=True,facecolor=fig.get_facecolor())  #self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
            self._save_fig("ir_pareto_customer_chart_normal",plot_output_dir,transparent=False,facecolor='black') 

        #    self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_customer_$_ranking",output_dir)
            plt.close()
        else:
            print("pareto customer nothing plotted. no records for ",par_df.shape)
            self.window.window_objects["working"]["winobject"].add_text("pareto customer nothing plotted. no records for "+str(par_df.shape))
 
        return




    def plot_pareto_product(self,par2_df,query_name,plot_output_dir):
        top=60
        new_df=par2_df.groupby(['product'],sort=False).sum()   #.copy()
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
        #    print("pareto product new_df=\n",new_df)
            fig, ax = pyplot.subplots()
            fig.patch.set_facecolor('None')
       #     fig.autofmt_xdate()
          #  ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places

#                ax.yaxis.set_major_formatter('${x:1.0f}')
          #  ax.yaxis.set_tick_params(which='major', labelcolor='green',
          #           labelleft=True, labelright=False)

         #   ax.ticklabel_format(style='plain') 
         #   ax.yaxis.set_major_formatter(ScalarFormatter())
      
            #ax.ticklabel_format(style='plain') 
      #      ax.axis([1, 10000, 1, 100000])
            
            ax=new_df.plot.bar(y='salesval',ylabel="",fontsize=7,grid=False,legend=False)
            ax.patch.set_facecolor('None')
        #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
       #     axis.set_major_formatter(ScalarFormatter())
         #   ax.ticklabel_format(style='plain')
            ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places

            ax.yaxis.set_tick_params(which='major', labelcolor='white',labelleft=True, labelright=False)

        #    ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" customer $ ranking total dollars "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
            ax.set_title(query_name[:105]+"\n"+query_name[105:205]+"\n"+query_name[205:300]+" $ sales product pareto",fontsize= 6,color='white')       
         
            ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=False,style=["w-"],secondary_y=True,legend=False)
            ax.xaxis.set_tick_params(labelcolor='white')
            ax2.xaxis.set_tick_params(labelcolor='white')
            ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
            ax2.yaxis.set_tick_params(labelcolor='white')
            ax3 = ax.twiny() 
            ax3.xaxis.set_tick_params(labelcolor='white')
            ax3.yaxis.set_tick_params(labelcolor='white')
            ax4=new_df[['pcount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
            if df_len<=1:
                df_len=2
     
            
            ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
            ax4.xaxis.set_tick_params(labelcolor='white')
            ax4.yaxis.set_tick_params(labelcolor='white')
            
            self._save_fig("ir_pareto_product_chart",plot_output_dir,transparent=True,facecolor=fig.get_facecolor())  #self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
            self._save_fig("ir_pareto_product_chart_normal",plot_output_dir,transparent=False,facecolor='black') 

        #    self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_customer_$_ranking",output_dir)
            plt.close()
        else:
            print("pareto product nothing plotted. no records for ",par2_df.shape)
            self.window.window_objects["working"]["winobject"].add_text("pareto product nothing plotted. no records for "+str(par2_df.shape))
 
        return





    def plot_yoy_dollars(self,qdf,query_name,plot_output_dir):
   #     print("yoy dollars plot type=",mat,latest_date.strftime('%d/%m/%Y'),output_dir)
   #     for k,v in df_dict.items():
   #         cust_df=self.preprocess(v,mat)
   #     cust_df=qdf.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        
        
        loffset = '7D'
        cust_df=qdf.resample('W-WED', label='left').sum().round(0)   
        cust_df.index = cust_df.index + to_offset(loffset) 
 
        
        

      #  print("end yoy customer preprocess",k)
        left_y_axis_title="$/week"
    
            
        year_list = cust_df.index.year.to_list()
       # week_list = cust_df.index.week.to_list()
        week_list = cust_df.index.isocalendar().week.to_list()
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
        fig.patch.set_facecolor('None')
        fig.autofmt_xdate()
      #  ax.ticklabel_format(style='plain')
        styles=["y-","r:","g:","m:","c:"]
        new_years=list(set(cust_df['year'].to_list()))
        #print("years=",years,"weels=",new_years)
        for y,i in zip(new_years[::-1],np.arange(0,len(new_years))):
            test_df=cust_df[cust_df['year']==y]
          #  print(y,test_df)
            fig=test_df[['salesval']].plot(use_index=True,grid=True,style=styles[i],lw=2,xlabel="",ylabel=left_y_axis_title,ax=ax,fontsize=8)
        
        ax.patch.set_facecolor('None')   
        
        ax.legend(new_years[::-1],fontsize=8,fancybox=True, framealpha=0.1,labelcolor='white')   #9
        
      #  leg = ax.get_legend()
      #  for i in len(leg.legendHandles):
       #     leg.legendHandles[i].set_color('white')

        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))   #,byweekday=mdates.SU)
#            ax.set_xticklabels([""]+improved_labels,fontsize=8)
        ax.set_xticklabels([""]+improved_labels,fontsize=7)

        ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places
        ax.xaxis.set_tick_params(labelcolor='white')
        ax.yaxis.set_tick_params(labelcolor='white')
        ax.set_title(query_name[:105]+"\n"+query_name[105:205]+"\n"+query_name[205:300]+"Year on Year $ sales / week",fontsize= 7,color='white')
  
     #   ax.yaxis.set_major_formatter('${x:1.0f}')
        ax.yaxis.set_tick_params(which='major', labelcolor='white',
                     labelleft=True, labelright=False)

        self._save_fig("ir_yoy_dollars_chart",plot_output_dir,transparent=True,facecolor=fig.get_facecolor())
        self._save_fig("ir_yoy_dollars_chart_normal",plot_output_dir,transparent=False,facecolor='black') 
        plt.close()
        return   
 



    def plot_salesval_activity_timeline(self,sales_df):
         x_offset=150
         y_offset=509
         new_df=sales_df.groupby(sales_df.index,sort=True).sum()  
    #     print("new_df=\n",new_df)
         new_df['days']=np.arange(1,new_df.shape[0]+1,1)+x_offset
         new_df['x_start']=np.full(new_df.shape[0],x_offset)
         new_df['y_start']=np.full(new_df.shape[0],y_offset)
 
         mat=new_df[['x_start','y_start','days','salesval']].tail(1200).to_numpy('int32')
      #   print("mat before =\n",mat.shape)
         maxyval=mat[:,3].max()
         xval=x_offset+10+new_df.shape[0]
         mat[:,3]=np.interp(mat[:,3], (mat[:,3].min(), mat[:,3].max()), (0, 95)).astype(np.int32)+y_offset
      #   print("mat after1 =\n",mat)
         
         pat_cords = tuple(mat.reshape(-1, mat.shape[0]*4)[0])
    #     print("cords =\n",cords)          

       #  cords = (0,0,100,100,0,0 , 120,100, 0,0,120,100, 0,0,120,150)
         howmany = len(pat_cords) / 4 # How many points not lines, not cordinates
         h=int(howmany)*2
    #     print("howmandy=",howmany,h)
         pat_colors = (255,255,0, 255,255,0)*int(howmany)   #int(howmany)
         return pat_cords,pat_colors,h,xval,maxyval
         
  


    
    
    def pivot_table_to_excel(self,df,plot_output_dir):
        new_sales_df=df.copy()
        new_sales_df["month"] = pd.to_datetime(new_sales_df["date"]).dt.strftime('%m-%b')
        new_sales_df["year"] = pd.to_datetime(new_sales_df["date"]).dt.strftime('%Y')
        
       # saved_sales_df=sales_df.copy(deep=True)
        
      #  saved_sales_df.replace({'productgroup':dd.productgroup_dict},inplace=True)
      #  sales_df.replace({'productgroup':dd.productgroups_dict},inplace=True)
      #  sales_df.replace({'specialpricecat':dd.spc_dict},inplace=True)
    
    
    ########################################################3
    
    
        pivot_df=pd.pivot_table(new_sales_df, values='salesval', index=['productgroup'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
       # pivot_df = pivot_df.reindex_axis(['year'], axis=1)
      #  pivot_df=pivot_df.T
    #    print("befoe=\n",pivot_df,pivot_df.columns)
        pivot_df.sort_index(axis='columns',level=["year","month"],ascending=[False,False], inplace=True, sort_remaining=False)
        pivot_df.drop("All",axis=1,level='year',inplace=True)
       # pivot_df = pivot_df.sort_index(('year', 'month'), ascending=False)
     #   print(pivot_df)
     #   pivot_df.sort_values(['year', 'month'], ascending=False,inplace=True)
     #   print(pivot_df)
    #    print("pod=",plot_output_dir)
    #    print("dddd",dd2.bbtv_dict['sales']['plot_output_dir'])
        pfilename="pivot_table_salesval.xlsx"
     #   print("pfilename=",pfilename)
        # dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
        # dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
                          
        sheet_name = 'Sheet1'
 
    #    writer = pd.ExcelWriter(dd2.bbtv_dict['sales']['plot_output_dir']+filename+".xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
        writer = pd.ExcelWriter(pfilename,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')

        pivot_df.to_excel(writer, sheet_name=sheet_name)
 
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        money_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': False})
        total_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': True})

        worksheet.set_column('B:ZZ', 12, money_fmt)
   #      worksheet.set_column('D:D', 12, total_fmt)
   #      worksheet.set_row(4, , total_fmt)

   #          # Apply a conditional format to the cell range.
   # #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
       # worksheet.conditional_format('B4:ZZ1000', {'type': '3_color_scale'})

        #value_fmt = workbook.add_format({'num_format': '#,##0.00', 'bold': False})
     # total_fmt = workbook.add_format({'num_format': '#,##0.00', 'bold': True})
 
   #     worksheet.set_column('D:ZZ', 12, value_fmt)
    #  worksheet.set_column('D:D', 12, total_fmt)
    #  worksheet.set_row(3, 12, total_fmt)
 
         # Apply a conditional format to the cell range.
#     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
  #    worksheet.conditional_format('D4:ZZ1000', {'type': '3_color_scale'})
 
        writer.save()      

        dfi.export(pivot_df.iloc[:,:25], "pivot_table_salesval.png", table_conversion='matplotlib',max_cols=26)
        im = Image.open("pivot_table_salesval.png").convert('RGB')
        im_invert = ImageOps.invert(im)
        im_invert.save("pivot_table_salesval.png")

        
        
        
        self.monthly_summary_sprite = pyglet.sprite.Sprite(pyglet.image.load('pivot_table_salesval.png'))
        self.monthly_summary_sprite.x=0
        self.monthly_summary_sprite.y=300
        self.monthly_summary_sprite.scale_y=0.74 #0.36
        self.monthly_summary_sprite.scale_x=0.66 #0.39
        self.monthly_summary_sprite.opacity=220
        self.monthly_summary_sprite_on=True

     #   ax = pivot_df.plot()
     #   fig = ax.get_figure()
     #   fig.savefig('pivot_df.png')








         
    
    def _save_fig(self,fig_id, output_dir,transparent,facecolor,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight',transparent=transparent,facecolor=facecolor, edgecolor='none')
        return
     
    
     
 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class text_box(object):   
    def __init__(self, window,startx,starty,tb_width,tb_height,*args, **kwargs):
        self.text_box_startx=startx  #startx
        self.text_box_starty=starty  #starty
        self.text_box_width=tb_width
        self.text_box_height=tb_height
     #   self.document = pyglet.text.document.FormattedDocument()
        self.document = pyglet.text.document.UnformattedDocument()  
        self.document.set_style(start=0,end=-1,attributes=dict(font_name='Arial', font_size=12, color=(255, 255, 255, 255)))

#        self.layout = pyglet.text.layout.IncrementalTextLayout(self.document, self.text_box_width, self.text_box_height,multiline=True,wrap_lines=True, batch=batch)
        self.layout = pyglet.text.layout.IncrementalTextLayout(self.document, self.text_box_width, self.text_box_height,multiline=True,wrap_lines=True, batch=batch)   #,attributes=dict(font_name='Arial', font_size=12, color=(255, 255, 255, 255)))
       # self.document.insert_text(0, " ")
        self.layout.x=self.text_box_startx
        self.layout.y=self.text_box_starty
        self.border_color=(randrange(255),100,randrange(255),255)
        self.total_text_length=0
        self.pageup_or_pagedown_jump=int(tb_height/CHAR_HEIGHT)-1
        self.mouse_over=False
        self.window=window
  
    
    def draw_text_box(self):  #,startx,starty,width,height): 
        self.layout = pyglet.text.layout.IncrementalTextLayout(self.document, self.text_box_width, self.text_box_height,multiline=True,wrap_lines=True, batch=batch)
        self.layout.x=self.text_box_startx
        self.layout.y=self.text_box_starty
  
    
    
    def draw_text_box_border(self):  
        tbatch = pyglet.graphics.Batch()
        pyglet.gl.glLineWidth(3)
        outline = tbatch.add(4, pyglet.gl.GL_LINE_LOOP, None, ('v2f', (self.layout.x, self.layout.y, self.layout.x+self.layout.width, self.layout.y, self.layout.x+self.layout.width, self.layout.y+self.layout.height, self.layout.x, self.layout.y+self.layout.height)), ('c4B', self.border_color*4))    #('c4B', (255, 0, 0, 0)*4))
        tbatch.draw()


    
    def add_text(self,text,*,font_size=12,color=(255, 255, 255, 255)):
        self.layout.view_y = self.layout.content_height
       
        self.layout.content_valign='bottom'
        self.document.insert_text(self.total_text_length, text+"\n", attributes=dict(font_size=font_size, color=color))
   
        #self.flip()
        self.window.flip()
        self.total_text_length+=len(text)      
        self.window.flip()
       # self.flip()
  
 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class bbtv_output_window(pyglet.window.Window,text_box,data_source):
    def __init__(self,width,height,*args,**kwargs):  #,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour,*args,**kwargs):
      super().__init__(*args,**kwargs)   #group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour,*args,**kwargs)
       # super().__init__()
      self.set_size(width, height)
       
      self.set_minimum_size(700,700)
      self.set_maximum_size(3048, 2048)
    
         # get window size
      self.x_max=self.get_size()[0]
      self.y_max=self.get_size()[1]
    
      self.x_temp=0
      self.y_temp=0
      
      self.window_objects={}
       #  def create_button_group(self,*,group_name,group_x_start,group_x_len,group_y_start,group_y_len,group_border_on,group_border_colour):
      # self.group_name=group_name
      # self.group_x_start=group_x_start
      # self.group_x_len=group_x_len
      # self.group_y_start=group_y_start
      # self.group_y_len=group_y_len
      # self.group_border_colour=group_border_colour
      # self.group_border_on=group_border_on
 
      
      
   #   print("bbtv sales data_source",data_source)
    
    
    
    
    def on_mouse_scroll(self,x, y, scroll_x, scroll_y):
       #  window.add_text("mouse scroll")
     
         for v in self.window_objects.values():
            b=v["winobject"] 
            if b.mouse_over:
          #  if self._check_for_mouse_over(x,y,tb.layout.x,tb.layout.y,tb.layout.x+tb.layout.width,tb.layout.y+tb.layout.height):
                if v['object_type']=="text_box": 
                    b.layout.view_y+=scroll_y*CHAR_HEIGHT
                    return
                elif v['object_type']=="button":     
                    b.list_move(-scroll_y,b,self.x_temp,self.y_temp)
                
    
    

    def on_key_press(self,symbol, modifiers):
        if keys[key.F1]:  
           for v in self.window_objects.values():
               if v['object_type']=="button":
                   b=v["winobject"]
               #    b.mouse_over=self._check_for_mouse_over(self.x_temp,self.y_temp,b.layout.x,b.layout.y,b.layout.x+b.layout.width,b.layout.y+b.layout.height)
                   if ((not b.floating) & b.active & b.visible & b.mouse_over):
                       if (not b.unique_list[b.unique_list_start_point] in b.selected_value_list) & (b.unique_list[b.unique_list_start_point]!=""):
                           if b.allow_multiple_selections:
                               b.selected_value_list.append(b.unique_list[b.unique_list_start_point]) 
                               
                            #   query_dict,query_df_list=st.create_query()
                            #   st.display_number_of_records(sales_df,query_dict,query_df_list,20,880,dd2.bbtv_dict['sales']['plot_output_dir'] ) 
                           else:
                               if len(b.selected_value_list)<=2:
                                   if len(b.selected_value_list)<=1:
                                       b.selected_value_list=[""] 
                                   elif len(b.selected_value_list)==2:
                                       del b.selected_value_list[-1]   #remove(b.unique_list[b.unique_list_start_point]) 
                                   b.selected_value_list.append(b.unique_list[b.unique_list_start_point]) 
                                 #  self.run_query()
                               #    query_dict,query_df_list=st.create_query()
                               #    st.display_number_of_records(sales_df,query_dict,query_df_list,20,880,dd2.bbtv_dict['sales']['plot_output_dir'] ) 
                                   return  
                               else:
                                   print("multiple selections not allowed on this field")
                               return    
 
                       
        elif keys[key.F2]:
            for v in self.window_objects.values():
                if v['object_type']=="button":
                     b=v["winobject"]
  
                     if ((not b.floating) & b.active & b.visible & b.mouse_over):
                         if b.unique_list[b.unique_list_start_point] in b.selected_value_list:               
                            if len(b.selected_value_list)>=2:                       
                                b.selected_value_list.remove(b.unique_list[b.unique_list_start_point]) 
                            #    query_dict,query_df_list=st.create_query()
                            #    st.display_number_of_records(sales_df,query_dict,query_df_list,20,880,dd2.bbtv_dict['sales']['plot_output_dir'] ) 
                                if b.name=='date_end':
                                   b.active=False
                                return
                            else:
                                print("selections empty")
                            return    
        elif keys[key.F3]:
            self.window_objects["working"]["winobject"].add_text("\n\n.....QUERYING...Please wait.\n\n")
         #   window_objects["working"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")  
         #   window_objects["results"]["winobject"].add_text("\nworking on query "+str(query_list)+"\n")       
            self.run_query()
              
            
        elif keys[key.UP]:
            for v in self.window_objects.values():
                b=v["winobject"]
              #    tb=v["winobject"]
                if b.mouse_over:
              #  if self._check_for_mouse_over(self.x_temp,self.y_temp,tb.layout.x,tb.layout.y,tb.layout.x+tb.layout.width,tb.layout.y+tb.layout.height):
                    if v['object_type']=="text_box": 
                        b.layout.view_y+=CHAR_HEIGHT
                    elif v['object_type']=="button":    
                        b.list_move(-1,b,self.x_temp,self.y_temp)
            
        
        elif keys[key.DOWN]:
            for v in self.window_objects.values():
                b=v["winobject"]
                 # tb=v["winobject"]
                if b.mouse_over:
               # if self._check_for_mouse_over(self.x_temp,self.y_temp,tb.layout.x,tb.layout.y,tb.layout.x+tb.layout.width,tb.layout.y+tb.layout.height):
                    if v['object_type']=="text_box": 
                        b.layout.view_y-=CHAR_HEIGHT
                    elif v['object_type']=="button":     
                        b.list_move(1,b,self.x_temp,self.y_temp)
            
         
        elif keys[key.PAGEUP]:
            for v in self.window_objects.values():
                b=v["winobject"] 
                  #tb=v["winobject"]
                if b.mouse_over:
                #if self._check_for_mouse_over(self.x_temp,self.y_temp,tb.layout.x,tb.layout.y,tb.layout.x+tb.layout.width,tb.layout.y+tb.layout.height):
                    if v['object_type']=="text_box": 
                        b.layout.view_y+=tb.pageup_or_pagedown_jump*CHAR_HEIGHT
                    elif v['object_type']=="button":     
                        b.list_move(-b.pageup_or_pagedown_jump,b,self.x_temp,self.y_temp)
            
            
        elif keys[key.PAGEDOWN]:
            for v in self.window_objects.values():
                b=v["winobject"]
                if b.mouse_over:
               # tb=v["winobject"] 
               # if self._check_for_mouse_over(self.x_temp,self.y_temp,tb.layout.x,tb.layout.y,tb.layout.x+tb.layout.width,tb.layout.y+tb.layout.height):
                    if v['object_type']=="text_box": 
                       b.layout.view_y-=b.pageup_or_pagedown_jump*CHAR_HEIGHT
                    elif v['object_type']=="button":    
                       b.list_move(b.pageup_or_pagedown_jump,b,self.x_temp,self.y_temp)
        
            #  self._list_move(window.list_page_move_length,x,y)  
        elif keys[key.ESCAPE]:
            self.close()
            return
        else:      
             #self._shortcut(pyglet.window.key.symbol_string(symbol),x,y)
               
            for v in self.window_objects.values():   # in zip(range(len(mouse_over_list)),text_box_list):
                b=v["winobject"]
                if b.mouse_over:
                #if check_for_mouse_over(mousex,mousey,tb.layout.x,tb.layout.y,tb.layout.x+tb.layout.width,tb.layout.y+tb.layout.height):
                    if v['object_type']=="text_box":  
                        b.add_text(pyglet.window.key.symbol_string(symbol))   #,tb1.document)
                    elif v['object_type']=="button": 
                        b._shortcut(pyglet.window.key.symbol_string(symbol),self.x_temp,self.y_temp,window_objects)
                        
                   
  
    
    def on_key_release(self, symbol, modifiers):       
        pass
    
    
  
    def on_mouse_enter(self,x, y):
        pass


    def on_mouse_leave(self,x, y):
        pass

  
    def on_mouse_press(self,x,y,mbutton, modifiers):
         if mbutton == mouse.LEFT:
            for v in self.window_objects.values():  
               if v['object_type']=="button":    # in zip(range(len(mouse_over_list)),text_box_list):
                   b=v["winobject"]
                   if ((not b.floating) & b.active & b.visible & b.pushable & b.mouse_over):  #x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                       if b.toggle:
                     #     b.marked_pushed=not b.marked_pushed
                        #  if not b.pushed:
                          #if not b.pushed:
                          b.pushed=not b.pushed
                          #else:
                          #   b.pushed=not b.pushed

                       else:
                      
                      #    print(b.name,b.pushed) 
                          b.pushed=True
                          if b.name=="query_sales_df":
                              self.window_objects["working"]["winobject"].add_text("\n\n.....QUERYING...Please wait.\n\n")
                              self.run_query()
                       #       print("run query now =true")
                       #       b.run_query_now=True
                      #    if not b.pushed:
                      #       b.pushed=True
                          elif b.name=="query_to_queue":
                              self.window_objects["query_queue"]["winobject"].add_text("QUERY: "+str(st.query_dict_save['query'])+" added to work queue.\n")
                          #    window_objects["query_queue"]["winobject"].add_text("QUERY: "+str(st.query_save_list)+" added to work queue.\n")
                              work_queue.append(st.query_save_list)
         elif mbutton == mouse.RIGHT:
           # print('The right mouse button was pressed.')
        #    if MY_DEBUG:
        #        text="the right mouse button was pressed. x="+str(x)+" y="+str(y)
            pass
 

 
    
    def on_mouse_drag(self,x, y, dx, dy, mbuttons, modifiers):
        if mbuttons & mouse.LEFT:
            for v in self.window_objects.values():  
               if v['object_type']=="button":  
                   b=v["winobject"]
                   if (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len)):
                       b.mouse_over=True
                       if b.floating:   
                           b.pushed=False
                        #   b.marked_pushed=False
                           b.x_start=x-self.x_offset
                           b.y_start=y-self.y_offset
                           b.layout.x=x-self.x_offset
                           b.layout.y=y-self.y_offset
             #       ir.move_and_draw_pointer(x,y)
                       elif (b.active & b.visible & b.movable & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                           b.floating=True 
                          
                           self.x_offset=x-b.x_start
                           self.y_offset=y-b.y_start
                  # else:
               #    b.floating=False
                  
          # print(text)
        elif mbuttons & mouse.RIGHT:
            pass
           # if MY_DEBUG:
           #     text="mouse drag right x,y,dx,dy"+str(x)+" "+str(y)+" "+str(dx)+" "+str(dy)
           #     ir._display_text_in_active_window(text)
    
        
        
        
        pass
 
    
    def on_mouse_release(self,x, y, button, modifiers):
    
        for v in self.window_objects.values():  
           if v['object_type']=="button":  
               b=v["winobject"]    
               if b.floating:   
                  b.floating=False

    
       
    def on_mouse_motion(self,x, y, dx, dy): 
        self.x_temp=x
        self.y_temp=y
         
        for v in self.window_objects.values():
             b=v["winobject"]
             b.mouse_over=self._check_for_mouse_over(x,y,b.layout.x,b.layout.y,b.layout.x+b.layout.width,b.layout.y+b.layout.height)
             if v["object_type"]=="button":  
                 if not b.mouse_over and not b.toggle:
                     b.pushed=False
                
  
    def update(self,dt):
#                stbg.draw_button_group_border()
    #    if stbg.querying_sign_on:
    #        pyglet.text.Label("Querying... Please wait.", font_name='Times New Roman',font_size=36,x=1000, y=600,batch=batch,color=(255,0,0,255))
    #        batch.draw()
    #    if self.querying_sign_on:
    #        pyglet.text.Label("Querying... Please wait.", font_name='Times New Roman',font_size=36,x=stbg.group_x_start+int(stbg.group_x_len/2)-100, y=stbg.group_y_start+int(stbg.group_y_len/2)+60,batch=batch,color=(255,0,0,255))
    #    self._draw_button_group_border

        pass

 
   
    def _check_for_mouse_over(self,mousex,mousey,winstartx,winstarty,winendx,winendy):
        return ((mousex>=winstartx and mousex<=winendx) and (mousey>=winstarty and mousey<=winendy))
 
    
 
    
 
    def run_query(self):
        # pyglet.clock.schedule_once(self._undraw_querying_sign, 6.0)
         window.flip()
         self.render()
         window.flip()
      #  time.sleep(1)
      #   window.flip()
      #   time.sleep(1)
     #    self._draw_querying_sign()
     #    window.flip()
       #  batch.draw()
      #   self._draw_button_group_border()
         query_dict,query_df_list=st.create_query()
         if len(query_dict)>0:
             self.window_objects["working"]["winobject"].add_text("\nworking on query "+str(query_dict['query'])+"\n")  
             self.window_objects["results"]["winobject"].add_text("\nworking on query "+str(query_dict['query'])+"\n")       
             st.display_number_of_records(sales_df,query_dict,query_df_list,20,880,dd2.bbtv_dict['sales']['plot_output_dir'] ) 
      #   pyglet.clock.unschedule(window._draw_querying_sign)
      #  pyglet.clock.schedule_once()
    #  self.querying_sign_on=False
       #  stbg.draw_button_group_border()
      #   self.querying_sign_on=False
      #   self._undraw_querying_sign()
         return
 
    
 
    
#     def _draw_button_group_border(self):      
#         pyglet.gl.glLineWidth(1)
#      #   print("1",data_source)
#     #    print("2",sales_buttons.data_source)
#        # print("3",sales)
# #        print("4",sales_buttons)
#       #  print("5",self.group_y_start)
#         outline = batch.add(4, pyglet.gl.GL_LINE_LOOP, None, ('v2f', (self.group_x_start, self.group_y_start, self.group_x_start+self.group_x_len, self.group_y_start, self.group_x_start+self.group_x_len, self.group_y_start+self.group_y_len, self.group_x_start, self.group_y_start+self.group_y_len)), ('c4B', self.group_border_colour*4))    #('c4B', (255, 0, 0, 0)*4))

    
 
    def _draw_report_sprites(self):
        result_sprites=[]
        if sales.pareto_product_sprite is not None:
           st.monthly_summary_sprite_on=False
           st.pareto_product_sprite.draw()
   #        result_sprites.append(pyglet.sprite.Sprite(st.pareto_product_sprite,batch=batch))
    #  self.mat_sprite = pyglet.sprite.Sprite(pyglet.image.load(plot_output_dir+'ir_mat_chart.png'))
    #                self.mat_sprite.x=1240
    #                self.mat_sprite.y=900
    #                self.mat_sprite.scale_y=0.5 #0.36
    #                self.mat_sprite.scale_x=0.55 #0.34
    #                self.mat_sprite.opacity=200
             #   
    #    result_sprites.append(pyglet.sprite.Sprite(plot_output_dir+'ir_pareto_product_chart.png',x=1240,y=900,batch=batch))
    #    result_sprites.update(scale_x=0.55,scale_y=0.5)
     #      plot_output_dir+'ir_pareto_product_chart.png'
                     
        if st.yoy_dollars_sprite is not None:
           st.yoy_dollars_sprite.draw()
     #      result_sprites.append(window.sprite.Sprite(st.yoy_dollars_sprite,batch=batch))
            
        if st.pareto_customer_sprite is not None:
            st.pareto_customer_sprite.draw()
     #       result_sprites.append(pyglet.sprite.Sprite(st.pareto_customer_sprite,batch=batch))
            
        if st.mat_sprite is not None:
           st.mat_sprite.draw()
      #     result_sprites.append(pyglet.sprite.Sprite(st.mat_sprite,batch=batch))
           
        if st.monthly_summary_sprite is not None and st.monthly_summary_sprite_on:
            st.monthly_summary_sprite.draw()
           #         result_sprites.append(pyglet.sprite.Sprite(self.pareto_customer_sprite, 210, 10, batch=batch))
           #         result_sprites.append(pyglet.sprite.Sprite(self.mat_sprite, 410, 10, batch=batch))
           #         result_sprites.append(pyglet.sprite.Sprite(self.yoy_dollars_sprite, 610, 10, batch=batch))
 
    
 
    def render(self):
        self.clear()
      #  if self.querying_sign_on:
      #      self._draw_querying_sign()
      #  self._draw_button_group_border()  
     #   self._draw_report_sprites()
        
    #    print("render windows_objkects",self.window_objects)
        for k,v in self.window_objects.items():
             if v["object_type"]=="text_box":
                 v["winobject"].draw_text_box_border()
              #   v["winobject"].add_text(k)
             elif v["object_type"]=="button" and v["winobject"].active:
                 v["winobject"].draw_button(k)  
                 if v["winobject"].mouse_over and v["winobject"].visible:
        #        #      v["winobject"].position_list_in_active_window(x=v["winobject"].x_start,y=v["winobject"].y_start-59,input_list=v["winobject"].unique_list[v["winobject"].unique_list_start_point:v["winobject"].unique_list_start_point+v["winobject"].unique_list_display_length],selection_list_input=True,color=(0,50,255,255))
        #        #      v["winobject"].position_list_in_active_window(x=self.x_temp-10,y=self.y_temp+len(v["winobject"].selected_value_list)*20+7,input_list=v["winobject"].selected_value_list,selection_list_input=False,color=(255,0,0,255))
                      v["winobject"].position_list_in_active_window(x=v["winobject"].x_start,y=v["winobject"].y_start-59,input_list=v["winobject"].unique_list[v["winobject"].unique_list_start_point:v["winobject"].unique_list_start_point+v["winobject"].unique_list_display_length],selection_list_input=True,color=(255,255,6,255))
                      v["winobject"].position_list_in_active_window(x=self.x_temp-10,y=self.y_temp+len(v["winobject"].selected_value_list)*20+7,input_list=v["winobject"].selected_value_list,selection_list_input=False,color=(255,255,6,255))
        # for k,v in window_objects.items():
        #     if v["object_type"]=="text_box":
        #         v["winobject"].draw_text_box_border()
        #      #   v["winobject"].add_text(k)
        #     elif v["object_type"]=="button" and v["winobject"].active:
        #         v["winobject"].draw_button(k)  
        #         if v["winobject"].mouse_over and v["winobject"].visible:
        #        #      v["winobject"].position_list_in_active_window(x=v["winobject"].x_start,y=v["winobject"].y_start-59,input_list=v["winobject"].unique_list[v["winobject"].unique_list_start_point:v["winobject"].unique_list_start_point+v["winobject"].unique_list_display_length],selection_list_input=True,color=(0,50,255,255))
        #        #      v["winobject"].position_list_in_active_window(x=self.x_temp-10,y=self.y_temp+len(v["winobject"].selected_value_list)*20+7,input_list=v["winobject"].selected_value_list,selection_list_input=False,color=(255,0,0,255))
        #              v["winobject"].position_list_in_active_window(x=v["winobject"].x_start,y=v["winobject"].y_start-59,input_list=v["winobject"].unique_list[v["winobject"].unique_list_start_point:v["winobject"].unique_list_start_point+v["winobject"].unique_list_display_length],selection_list_input=True,color=(255,255,6,255))
        #              v["winobject"].position_list_in_active_window(x=self.x_temp-10,y=self.y_temp+len(v["winobject"].selected_value_list)*20+7,input_list=v["winobject"].selected_value_list,selection_list_input=False,color=(255,255,6,255))
  
        batch.draw()
        return
        
      
        
    
    def on_draw(self):
        self.render()        
         
        dt=clock.tick()
        if MY_DEBUG:
            self.position_text_in_active_window("fps="+str(int(clock.get_fps()))+" size="+str(window.get_size())+" loc="+str(window.get_location())+" Pos=("+str(x)+","+str(y)+") dx=("+str(dx)+","+str(dy)+")",x=0,y=5)
            
       # batch.draw()
   
    
        
      
      
      
  

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





class gui(object):
    def __init__(self):
       pass
    
    def report_query_info(self,df,window_objects):
    
      #  window_objects["working"]["winobject"].add_text("Most recent 40 trans - "+df.head(40).to_string()+"\n")
        window_objects["working"]["winobject"].add_text("\n"+str(df.shape)) 
        
        window_objects["results"]["winobject"].add_text("\nTotal Sales Transactions size="+'{:,.0f}'.format(df.shape[0])+" rows\n")
        if df.shape[0]>0:
         #   print(df)
            total_sales=df['salesval'].sum()
            total_qty=int(df['qty'].sum())
            window_objects["results"]["winobject"].add_text("First date:"+str(df.index[-1])+", Last date:"+str(df.index[0])+"\n")
            window_objects["results"]["winobject"].add_text("Total Sales Transactions for query NSV value"+' ${:,.2f}'.format(total_sales)+"\n")
            window_objects["results"]["winobject"].add_text("Total Sales for query Unit Qty days="+'{:,.0f}'.format(total_qty)+"\n")
          
            new_df=df.groupby(['period'],sort=True).sum()
        #    print(new_df)
        #    print("First date:"+str(new_df.index[-355])+", Last date:"+str(new_df.index[-1])+"\n")
             
            total_sales=new_df['salesval'].iloc[-355:].sum()
            total_qty=int(new_df['qty'].iloc[-355:].sum()) 
            
            window_objects["results"]["winobject"].add_text("\nFirst date:"+str(new_df.index[-355])+", Last date:"+str(new_df.index[-1]))
            window_objects["results"]["winobject"].add_text("\nTotal Sales Transactions NSV value last 365 days"+' ${:,.2f}'.format(total_sales)+"\n")
            window_objects["results"]["winobject"].add_text("Total Sales Unit Qty last 365 days="+'{:,.0f}'.format(total_qty)+"\n")
        else:
            window_objects["results"]["winobject"].add_text("sales df empty.\n")
        return window_objects
    
    
    def display_win_objects(self,window_objects):
       for k,v in window_objects.items():
         #  print(k)
           window_objects["working"]["winobject"].add_text("key"+str(k)+"\n")
           for i,j in v.items():
          #    print("    ",i,":",j)
              window_objects["working"]["winobject"].add_text("key"+str(i)+"="+str(j)+"\n")
    
    
    def display_start_banner(self,window):
           
     #    print("\nBBTV1-Beerenberg analyse, visualise and predict- By Anthony Paech 26/1/2021")
     #    print("=================================================================================================\n")       
        
     #    print("Python version:",sys.version)
     #    print("Current working directory",cwdpath)
     #    print("plot_output dir",plot_output_dir)
     #    print("data save directory",dd2.bbtv_dict['sales']['save_dir'])
     # #   print("data save directory",dd2.bbtv_dict['sales']['plot_output_dir'])
     #    print("\ntensorflow:",tf.__version__)
     #    #print("TF2 eager exec:",tf.executing_eagerly())      
     #    print("keras:",keras.__version__)
     #    print("numpy:",np.__version__)
     #    print("pandas:",pd.__version__)
     #    print("pyglet:",pyglet.version)
     #    print("pyodbc:",pyodbc.version)
     #    print("matplotlib:",mpl.__version__)      
     #    print("sklearn:",sklearn.__version__)         
     #    print("\nnumber of cpus : ", multiprocessing.cpu_count())            
     #    print("tf.config.get_visible_devices('GPU'):\n",visible_devices)
     # #   print("\n",pd.versions(),"\n")
     #    print("\n=================================================================================================\n")     
        
        window.window_objects["working"]["winobject"].add_text("\n\nBBTV1-Beerenberg analyse, visualise and predict- By Anthony Paech 26/1/2021"+"\n")
        window.window_objects["working"]["winobject"].add_text("=================================================================================================\n")
        window.window_objects["working"]["winobject"].add_text("Python version:"+str(sys.version)+"\n")
        window.window_objects["working"]["winobject"].add_text("Current working directory:"+str(cwdpath)+"\n")
        window.window_objects["working"]["winobject"].add_text("plot_output dir:"+str(plot_output_dir)+"\n")
        window.window_objects["working"]["winobject"].add_text("data save directory:"+str(dd2.bbtv_dict['sales']['save_dir'])+"\n")
        window.window_objects["working"]["winobject"].add_text("\ntensorflow:"+str(tf.__version__)+"\n")
        window.window_objects["working"]["winobject"].add_text("keras:"+str(keras.__version__)+"\n")
        window.window_objects["working"]["winobject"].add_text("numpy:"+str(np.__version__)+"\n")
        window.window_objects["working"]["winobject"].add_text("pandas:"+str(pd.__version__)+"\n")
        window.window_objects["working"]["winobject"].add_text("pyglet:"+str(pyglet.version)+"\n")
        window.window_objects["working"]["winobject"].add_text("pyodbc:"+str(pyodbc.version)+"\n")
        window.window_objects["working"]["winobject"].add_text("matplotlib:"+str(mpl.__version__) +"\n")
        window.window_objects["working"]["winobject"].add_text("sklearn:"+str(sklearn.__version__)+"\n")
        window.window_objects["working"]["winobject"].add_text("\nnumber of cpus : "+str(multiprocessing.cpu_count())+"\n")
        window.window_objects["working"]["winobject"].add_text("tf.config.get_visible_devices('GPU'):\n"+str(visible_devices)+"\n")
        window.window_objects["working"]["winobject"].add_text("\n")
        window.window_objects["working"]["winobject"].add_text("\n")
        window.window_objects["working"]["winobject"].add_text("\n")
        return
    
    
    
      
        
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
 
        
 
        
    
    def gui_queries(self,sales_df):
       # bbtv1=bbtv1_root.bbtv1_class()
      #  print("in",sys.argv)
      #  global window
    #        global window_objects
    #        global window_objects
        
         
            
    #         global window_objects
    #         window_objects={}
    #         window_objects["results"]={"object_type":"text_box","winobject":text_box(startx=0,starty=0,width=2300,height=300)}
    #         window_objects["working"]={"object_type":"text_box","winobject":text_box(startx=2300,starty=0,width=700,height=1800)}
    #         window_objects["instructions"]={"object_type":"text_box","winobject":text_box(startx=0,starty=1800,width=600,height=200)}
    #         window_objects["query_queue"]={"object_type":"text_box","winobject":text_box(startx=600,starty=1800,width=2400,height=200)}
        
    #         window_objects["results"]["winobject"].add_text("Results window ")
    #         window_objects["working"]["winobject"].add_text("Working window ")
    #         window_objects["instructions"]["winobject"].add_text("Instructions ")
    #         window_objects["query_queue"]["winobject"].add_text("Query queue window \n")
        
    # #        global sales_df=dd2.dash2_dict 
     #   global work_queue
        
    
    
        
       # print("sales_df=\n",sales_df)
        #bg=button_group()
        #b=button()
        #os.chdir("/home/tonedogga/Documents/python_dev")
        
        
        
        ##########################################3
        # if ODBC fails
        load_file="raw_savefile.pkl"
        load_dir="./bbtv1_saves/"
        
        load_production_dir="./"
        load_production_file="production_df.pkl"
        
        ############################################
        
        button_startx=50
        button_starty=1600
        work_queue=[]
  
         
     #   sales_button=button()
     #   sales_buttons=buttons()
         
      #  sales=data_source()


        config = pyglet.gl.Config(double_buffer=True) 
        window=bbtv_output_window(width=3000,height=2000,resizable=False,visible=False,caption="BBTV ver 1.0 - Beerenberg analyse, visualise and predict",config=config)    #,group_name="sales_trans",group_x_start=0,group_x_len=2300,group_y_start=1500,group_y_len=300,group_border_on=True,group_border_colour=(145,234,45,255))
        window.push_handlers(keys) 
           
       # print("wwo",window.window_objects)    
      
        sales=data_source(window=window,group_name="sales_trans",group_x_start=0,group_x_len=2300,group_y_start=1500,group_y_len=300,group_border_on=True,group_border_colour=(145,234,45,255))
    

    #    sales_buttons=buttons(group_name="sales_trans",group_x_start=0,group_x_len=2300,group_y_start=1500,group_y_len=300,group_border_on=True,group_border_colour=(145,234,45,255))
 
      #  x,f=sales.load_pickle("./sales_df.pkl")
#        sales.data_source_box(height=33,width=44)
        sales.pivot_table_to_excel(sales_df,plot_output_dir)
        unique_dict=sales.find_uniques(sales_df)
       # print("unuqie dict",unique_dict)

        
      #  sales_buttons.create_button_group(group_name="sales_trans",group_x_start=0,group_x_len=2300,group_y_start=1500,group_y_len=300,group_border_on=True,group_border_colour=(145,234,45,255))
   
        sales.create_buttons(button_startx,button_starty,'./buttons.csv',sales.sales_df_dict,unique_dict)
         
         

        results=text_box(window=window,startx=0,starty=0,tb_width=2300,tb_height=300)
        working=text_box(window=window,startx=2300,starty=0,tb_width=700,tb_height=1800)
        instructions=text_box(window=window,startx=0,starty=1800,tb_width=600,tb_height=200)
        query_queue=text_box(window=window,startx=600,starty=1800,tb_width=2400,tb_height=200)
 
        window.window_objects["results"]={"object_type":"text_box","winobject":results}   #text_box(startx=0,starty=0,width=2300,height=300)}
        window.window_objects["working"]={"object_type":"text_box","winobject":working}   #text_box(startx=2300,starty=0,width=700,height=1800)}
        window.window_objects["instructions"]={"object_type":"text_box","winobject":instructions}     #text_box(startx=0,starty=1800,width=600,height=200)}
        window.window_objects["query_queue"]={"object_type":"text_box","winobject":query_queue}      #text_box(startx=600,starty=1800,width=2400,height=200)}


        print("wwo",window.window_objects) 
      #  sales.window(group_name="sales_trans",group_x_start=0,group_x_len=2300,group_y_start=1500,group_y_len=300,group_border_on=True,group_border_colour=(145,234,45,255))
     #   print("sb",sales_buttons)
    #    window.results=text_box(startx=0,starty=0,tb_width=2300,tb_height=300)
        results.add_text("Results window")
        working.add_text("Working window ")  
        instructions.add_text("Instructions window ")  
        instructions.add_text("\nTo Select:Mouse over the fields rectangle, \nthen PageUp/Pagedown/up/down arrows/mouse scroll wheel or type first letter. \nSelect start date and end first.\nPress F1-add a selection, F2-remove the selected value. \nF3 or [query] button to query on current selection. Esc to exit.")
        query_queue.add_text("query queue window ")
     #   print(sales)
    #    results.display_text_box()
 
    #    window.display_text_box(working)
    #    window.display_text_box(results)
    #    window.display_text_box(query_queue)       
    #    window.display_text_box(instructions)
        self.display_start_banner(window)
        
        window.set_visible()
        pyglet.app.run()
           
            
        window.close()
        print("wq=",work_queue)
        return work_queue    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#         config = pyglet.gl.Config(double_buffer=True)      
#         window = bbtv_output_window(width=3000,height=2000,wobjects=window_objects,resizable=False,visible=False,caption="BBTV ver 1.0 - Beerenberg analyse, visualise and predict",config=config)
#         window.push_handlers(keys) 
    
#       #  window_objects=b.setup_buttons(window_objects,st.sales_df_dict,unique_dict)
#      #   print(windows_objects)
#       #  print(json.dumps(windows_objects, indent = 4))
#       #  wd="windows objects"+windows_objects
#       #  window_objects["working"]["winobject"].add_text(wd)
          
     
#         #display_win_objects(window_objects)
    
#        # pyglet.clock.schedule_interval(window.update, 0.1)
#         window.set_visible()
#         pyglet.app.run()
       
        
#         window.close()
#        # print("wq=",work_queue)
#         return work_queue
    
     
 



# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





gui=gui()

# sales_button=button()   #,"syp",33,44)
# buttons("sales but1",7)

# # #stbg=button_group()
# sales_button.create_button_group(group_name="sales_trans",group_x_start=0,group_x_len=2300,group_y_start=1500,group_y_len=300,group_border_on=True,group_border_colour=(145,234,45,255))


# sales=data_source("sales",9,8)
# x,f=sales.load_data("sales_df")
# sales.data_source_box(height=33,width=44)

# button("xp","yp",3,4)
# production_buttons=buttons("prod but1",7)

# production=data_source("production",19,18)
# x,f=production.load_data("production_df")
# production.data_source_box(height=55,width=66)

# #results=text_box("mresults",300,50)
# #working=text_box("wmorking",30,5)
# window_objects={}
# window_objects["results"]={"object_type":"text_box","winobject":text_box(startx=0,starty=0,width=2300,height=300)}
# window_objects["working"]={"object_type":"text_box","winobject":text_box(startx=2300,starty=0,width=700,height=1800)}
# window_objects["instructions"]={"object_type":"text_box","winobject":text_box(startx=0,starty=1800,width=600,height=200)}
# window_objects["query_queue"]={"object_type":"text_box","winobject":text_box(startx=600,starty=1800,width=2400,height=200)}

# window_objects["results"]["winobject"].add_text("Results window ")
# window_objects["working"]["winobject"].add_text("Working window ")
# window_objects["instructions"]["winobject"].add_text("Instructions ")
# window_objects["query_queue"]["winobject"].add_text("Query queue window \n")


# window=bbtv_output_window(100,200,300)
# window.display_window(height=888,width=999)


# production.display_button("j")
# production.display_buttons("k")
# sales.display_button("l")
# sales.display_buttons("m")


# window.display_text_box(working)
# window.display_text_box(results)


# window.set_visible()
# pyglet.app.run()
   
    
# window.close()
# print("wq=",work_queue)
    



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
