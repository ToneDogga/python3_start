#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:10:04 2020

@author: tonedogga
"""
import string
import numbers

import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path
from p_tqdm import p_map,p_umap

import datetime as dt

import dash2_dict as dd2

import pyglet
from pyglet import clock
from pyglet import gl
from pyglet.gl import *

from pyglet.window import key
from pyglet.window import mouse

from pyglet import shapes
from pyglet import image
#pic = image.load('picture.png')

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


#from time import time

MY_DEBUG=False #True   #False
BUTTON_LIST=[]
#BUTTON_COLOR=(200,100,200)
#global batch
load_file="raw_savefile.pkl"
load_dir="./dash2_saves/"


#########################################################################################################################




class QueryWindow(pyglet.window.Window):
    def __init__(self,*args,**kwargs):
     #   super(MyWindow,self).__init__(*args,**kwargs)
        super(QueryWindow,self).__init__(*args,**kwargs)
 
        #set window size
        self.set_minimum_size(700,700)
        self.set_maximum_size(2048, 2048)
        
        # get window size
        self.x_max=self.get_size()[0]
        self.y_max=self.get_size()[1]

       # draw_buttons_to_batch()
     #   self.moved=False
        self.list_page_move_length=29
        #print(self.get_size())
        self.x_temp=0
        self.y_temp=0
        # get window location
        #x, y = window.get_location()
        #window.set_location(x + 20, y + 20) 
 
    # button_list is global    
 
    #  buttons=get_button_details()
         
    def on_key_press(self, symbol, modifiers):
        act_on_key_state(self.x_temp,self.y_temp,symbol,modifiers)
        # if symbol == key.F1:
        #    for b in BUTTON_LIST:
        #       if ((not b.floating) & b.active & b.visible & b.mouse_over):
        #           if (not b.unique_list[b.unique_list_start_point] in b.selected_value_list) & (b.unique_list[b.unique_list_start_point]!=" "):
        #               b.selected_value_list.append(b.unique_list[b.unique_list_start_point]) 
        #               query_dict=st.create_query()
        # elif symbol == key.F2:
        #     for b in BUTTON_LIST:
        #        if ((not b.floating) & b.active & b.visible & b.mouse_over):
        #           if b.unique_list[b.unique_list_start_point] in b.selected_value_list:
        #               b.selected_value_list.remove(b.unique_list[b.unique_list_start_point]) 
        #               query_dict=st.create_query()
        # elif symbol == key.UP:
        #     self.moved=self._list_move(1)
        # elif symbol == key.DOWN:
        #     self.moved=self._list_move(-1)
        # elif symbol == key.PAGEUP:
        #     self.moved=self._list_move(self.list_page_move_length)
        # elif symbol == key.PAGEDOWN:
        #     self.moved=self._list_move(-self.list_page_move_length)      
        # else:        
        #     self.moved=self._shortcut(pyglet.window.key.symbol_string(symbol))
        # return 
        pass
    
    
    def on_key_release(self, symbol, modifiers):
        
        pass
    
    
  
    def on_mouse_enter(self,x, y):
        pass

    def on_mouse_leave(self,x, y):
        pass
    
    def on_mouse_motion(self,x, y, dx, dy):
      #  fps_display.draw()
      #  batch=check_for_mouse_over(x,y)
        move_and_draw_pointer(x,y)
        self.x_temp=x
        self.y_temp=y
       
   
 
    def on_mouse_release(self,x, y, button, modifiers):
        for b in BUTTON_LIST:
            if b.floating:   
                b.floating=False
                
        


    def on_mouse_drag(self,x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT:
            if MY_DEBUG:
                text="mouse drag left x,y,dx,dy"+str(x)+" "+str(y)+" "+str(dx)+" "+str(dy)
                _display_text_in_active_window(text)
            for b in BUTTON_LIST:
                if b.floating:   
                    b.pushed=False
                    b.x_start=x-self.x_offset
                    b.y_start=y-self.y_offset
                    move_and_draw_pointer(x,y)
                elif (b.active & b.visible & b.movable & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                    b.floating=True 
                    self.x_offset=x-b.x_start
                    self.y_offset=y-b.y_start
          # print(text)
        elif buttons & mouse.RIGHT:
            if MY_DEBUG:
                text="mouse drag right x,y,dx,dy"+str(x)+" "+str(y)+" "+str(dx)+" "+str(dy)
                _display_text_in_active_window(text)
          # print(text)
  
     
  
    
   
    def on_mouse_press(self,x,y,button, modifiers):
        if button == mouse.LEFT:
             for b in BUTTON_LIST:
                if ((not b.floating) & b.active & b.visible & b.pushable & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                    b.pushed=not b.pushed
  
        elif button == mouse.RIGHT:
           # print('The right mouse button was pressed.')
            if MY_DEBUG:
                text="the right mouse button was pressed. x="+str(x)+" y="+str(y)

                _display_text_in_active_window(text)
    
    
  
    
  
    
    def on_mouse_scroll(self,x, y, scroll_x, scroll_y):
     #   for b in BUTTON_LIST:
     #       b.mouse_over=False
     #       if ((not b.floating) & b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
           #     b.mouse_over=True
                _list_move(-scroll_y,x,y)
                # if (b.unique_list_start_point>=1) & (b.unique_list_start_point<=len(b.unique_list)-1):
                #     b.unique_list_start_point+=scroll_y
                #     if b.unique_list_start_point<1:
                #         b.unique_list_start_point=1
                #     if b.unique_list_start_point>len(b.unique_list)-1:
                #         b.unique_list_start_point=len(b.unique_list)-1
                #   #  b.selected_value_list.append(b.unique_list[b.unique_list_start_point])    
                #    position_list_in_active_window(x=x,y=y,input_list=b.unique_list[b.unique_list_start_point:])
                move_and_draw_pointer(x,y)
   #     if MY_DEBUG:
   #         text="mouse scroll x,y,scroll_x,scroll_y"+str(x)+" "+str(y)+" "+str(scroll_x)+" "+str(scroll_y)
   #         _display_text_in_active_window(text)

        
   

    def on_draw(self):  
        #self.clear()
      
        draw_buttons()
   #     check_for_mouse_over(x,y)
        pass
 
 
    
 #   def on_resize(self,width, height):
   #     print('The window was resized to %dx%d' % (width, height))
      #  display = pyglet.canvas.Display()
      #  screen = display.get_default_screen()
      #  self.screen_width = screen.width
      #  self.screen_height = screen.height

      #  self.clear()
  #      pass
    
    def update(self,dt):
    #    display_buttons()
        #draw_batch(x,y,dx,dy)
    #    window.clear()
        pass

        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class sales_trans(object):
    def __init__(self):
        self.sales_df_dict={   # fields to display
            "date":2,  # two date fields, date_start, date_end
            
            "cat":1,
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
        
        self.qdf_size=0
        self.qdf_sales=0
        self.qdf_qty=0
        self.display=False
        self.query_dict_save={}
        self.display_x=30
        self.display_y=1000
        self.first_date_needed=True
        self.second_date_needed=False
        
        self.xy=zip(np.array([0]),np.array([0]))
        self.saved_xy=zip(np.array([0]),np.array([0]))
        
        
    
    def load_pickle(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            print("\nLoading "+str(save_dir)+str(savefile)+"...")
            return self.preprocess(pd.read_pickle(save_dir+savefile))
        else:
            print("load sales_df error.")
            return
        
    
    def preprocess(self,df):    #, rename_dict):
     #   print("preprocess_sc sales save df=",df,rename_dict)
        df=df[(df['code']!="OFFINV")]   
        df=df[(df['product']!="OFFINV")]   
        
        df=df[(df['product']!=0)]   
        
     #   df.to_pickle(dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['raw_savefile'],protocol=-1)
 
    #---------------------------------------------------------
 
       # if dd2.dash2_dict['sales']['glset_not_spc_mask_flag']:
       #     new_df=df[((df['productgroup'].isin(dd2.dash2_dict['sales']['pg_only'])) & (df['glset'].isin(dd2.dash2_dict['sales']['glset_only'])))].copy()  
       # else:    
     #   if dd2.dash2_dict['sales']['apply_mask']:  # apply masking                                                                                          
     #       df=df[((df['productgroup'].isin(dd2.dash2_dict['sales']['pg_only'])) & (df['specialpricecat'].isin(dd2.dash2_dict['sales']['spc_only'])))]  

        new_df=df[~df['product'].isin(dd2.dash2_dict['sales']["GSV_prod_codes_to_exclude"])].copy()    
 
    
#------------------------------------------------------------------------


        print("\nPreprocess data to convert GSV to NSV exclude products=",dd2.dash2_dict['sales']["GSV_prod_codes_to_exclude"])
       # df=df[(df['code']=="OFFINV") | (df['product']=="OFFINV")]
    #    new_df.rename(columns=rename_dict, inplace=True)
     #   df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        #print("rename collumsn"
    #    print("preprocesed=\n",new_df.head(100))
        return new_df
   


       
    
    def _save_fig(self,fig_id, output_dir,transparent,facecolor,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight',transparent=transparent,facecolor=facecolor, edgecolor='none')
        return
 




    def find_uniques(self,sales_df):
       # position_text_in_active_window("Sales Transactions size="+str(sales_df.shape[0])+" rows",x=0,y=0)
        print("Indexing selection columns...")
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
        print("Ready.\n")
        return unique_dict


#     def _graph_mat(self,xy):   #mat,x_vals,y_vals):
        
        
#         batch = pyglet.graphics.Batch()
#        # _draw_rect(x_start,y_start,x_start+x_np.max(),y_start+y_np.max(),colour=(255,255,255),batch=batch)
            
#         for x,y in xy:   #zip(x_vals,y_vals):
#             batch.add(2, pyglet.gl.GL_POINTS, None,
#                                   ('v2i', (0,10,x, y)),             
#                                   ('c3B', (255,255,255,255, 255, 255))
#                                   )
    
    
# #     vertex_list = batch.add(2, pyglet.gl.GL_POINTS, None,
# #     ('v2i', (10, 15, 30, 35)),
# #     ('c3B', (0, 0, 255, 0, 255, 0))
# # )
    
    
# #         batch.add(2, pyglet.gl.GL_LINES, None,
# #                                   ('v2i', (0, y_max, x, y)),             
# #                                   ('c3B', (0, 255, 0, 255, 255, 255))
# #         )
        
# #         batch.add(2, pyglet.gl.GL_LINES, None,
# #                                   ('v2i', (x_max,0, x, y)),             
# #                                   ('c3B', (0, 0, 255, 255, 255, 255))
# #         )
    
# #         batch.add(2, pyglet.gl.GL_LINES, None,
# #                                   ('v2i', (x_max, y_max, x, y)),             
# #                                   ('c3B', (60, 70, 20, 255, 255, 255))
# #         )
#         batch.draw() 
#         return  




    # def quick_mat(self,query_df_list,x,y,x_offset=0,y_size=500):
    #     #  draw an mat in pyglet!
    #     #
    #  #  self.saved_xy=zip(np.array([0]),np.array([0]))
    #     if len(query_df_list)>0:
    #         df=query_df_list[0].copy()
    #     #    print("df shape",df.shape)
    #         new_df=df.groupby(['period'],sort=True).sum()
    #         new_df['mat']=new_df['salesval'].rolling(365,axis=0,closed='left').sum().round(0)
    #     #    new_df['days']=np.arange(-new_df.shape[0]+1,1,1)
    #         new_df['days']=np.arange(1,new_df.shape[0]+1,1)
           
    #         mat=new_df.iloc[365:][['days','mat']].to_numpy('int32')
    #         self.width=mat.shape[0]
    #      #   print("bnew_df=\n",mat) 
    #         x_max=window.x_max
    #         y_max=window.y_max
 
    #  #   batch.add(2, pyglet.gl.GL_LINES, None,
    #   #                           ('v2i', (0, 0, x, y,0,y_max)),
    #                          #    ('v2i', (x_max,0, x, y,x_max,y_max)), 
    #   #                           ('c3B', (255, 0, 0, 255, 255, 255))
    #   #  )
      
    #         x_vals=mat[:,0]+x_offset+x  
    #         y_vals=mat[:,1]+y
        
    #    # print("yvals",y_vals)
    #         scaled_y_vals=np.interp(y_vals, (y_vals.min(), y_vals.max()), (0, y_size)).astype(np.int32)
    #       #  print("scaled_y_vals",scaled_y_vals,x_vals)
    #         xy=zip(x_vals,scaled_y_vals)
    #         if (xy!=None):
    #           #  if (xy!=self.saved_xy): 
    #                 self.saved_xy=xy
    
    #    # else:
    #     if True:
    #         batch = pyglet.graphics.Batch()
     
    #         for new_x,new_y in self.saved_xy:   #zip(x_vals,y_vals):
    #             batch.add(2, pyglet.gl.GL_POINTS, None,
    #                                       ('v2i', (x,y,new_x, new_y)),             
    #                                       ('c3B', (255,255,255,255, 255, 255))
    #                                   )
        
    #         batch.draw()
    #         _draw_rect(x,y-y_size,self.width,y_size,(255,0,0),batch)
    #     return



  #  sprite = pyglet.sprite.Sprite(pyglet.image.load('./roar.png'))
  #  sprite.draw()



    def display_number_of_records(self,sales_df,query_dict,query_df_list,x,y):
        
        print_flag=False
     #   pyglet.TextLayout.color=(255,255,255,255) 
        position_text_in_active_window("To Select:Mouse over the fields rect, then PageUp/Pagedown/up/down arrows/mouse scroll wheel or type first letter. Press F1-add a selection, F2-remove the selected val. Esc to exit.",x=x,y=y)
        position_text_in_active_window("Total Sales Transactions size="+'{:,.0f}'.format(sales_df.shape[0])+" rows",x=x,y=y-15)
        self.total_sales=sales_df['salesval'].sum()
        self.total_qty=int(sales_df['qty'].sum())
        position_text_in_active_window("Total Sales Transactions NSV value "+'${:,.2f}'.format(self.total_sales),x=x,y=y-30)
        position_text_in_active_window("Total Sales Unit Qty="+'{:,.0f}'.format(self.total_qty),x=x,y=y-45)
        
      #  if len(query_df_list)>0:
        #    print("query df list=\n",query_df_list)   
        if len(query_df_list)==1:
            qdf=query_df_list[0]
            if self.qdf_size!=qdf.shape[0]: 
                self.qdf_size=qdf.shape[0]
                print("Query complete",self.qdf_size,"records") 
                self.qdf_sales=qdf['salesval'].sum()
                self.qdf_qty=int(qdf['qty'].sum())
                self.display=True
                self.plot_mat(qdf,str(query_dict))
                self.mat_sprite = pyglet.sprite.Sprite(pyglet.image.load('./mat_chart.png'))
                self.mat_sprite.x=800
                self.mat_sprite.y=20
                self.mat_sprite.scale_y=0.5
                self.mat_sprite.scale_x=0.4
                self.mat_sprite.opacity=200
                
                self.plot_pareto_customer(qdf,str(query_dict))
                self.pareto_customer_sprite = pyglet.sprite.Sprite(pyglet.image.load('./pareto_customer_chart.png'))
                self.pareto_customer_sprite.x=10
                self.pareto_customer_sprite.y=20
                self.pareto_customer_sprite.scale_y=0.5
                self.pareto_customer_sprite.scale_x=0.45
                self.pareto_customer_sprite.opacity=200

                self.plot_pareto_product(qdf.copy(),str(query_dict))
                self.pareto_product_sprite = pyglet.sprite.Sprite(pyglet.image.load('./pareto_product_chart.png'))
                self.pareto_product_sprite.x=10
                self.pareto_product_sprite.y=20
                self.pareto_product_sprite.scale_y=0.5
                self.pareto_product_sprite.scale_x=0.45
                self.pareto_product_sprite.opacity=200


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
            position_text_in_active_window("Query="+query_name[:170],x=x,y=y-65)
            position_text_in_active_window(query_name[170:350],x=x,y=y-80)
            position_text_in_active_window(query_name[350:],x=x,y=y-95)



            for c in BUTTON_LIST:
            
                if (c.name=="date_start") & (len(c.selected_value_list)>1):
                    start_date=pd.to_datetime(c.selected_value_list[-1])
               #     print(start_date)
                if (c.name=="date_end") & (len(c.selected_value_list)>1):
                    end_date=pd.to_datetime(c.selected_value_list[-1])
                #    print(end_date)
            if isinstance(start_date,pd.Timestamp) & isinstance(end_date,pd.Timestamp):  
                query_span=(end_date-start_date).days
                        
                position_text_in_active_window("Query day length="+str(query_span)+" days",x=x,y=y-115)
            position_text_in_active_window("Queried Sales Transactions size="+'{:,.0f}'.format(self.qdf_size)+" rows",x=x,y=y-130)
            position_text_in_active_window("Query Transactions NSV value "+'${:,.2f}'.format(self.qdf_sales),x=x,y=y-145)
            position_text_in_active_window("Query Unit Qty="+'{:,.0f}'.format(self.qdf_qty),x=x,y=y-160)
            
         #   self.sprite.x=x
          #  self.sprite.y=y-500
       #     batch = pyglet.graphics.Batch()
            self.mat_sprite.draw()
            for c in BUTTON_LIST:
                if c.name=="pareto_type":
                    if c.pushed:   
                        self.pareto_product_sprite.draw()
                    else:
                        self.pareto_customer_sprite.draw()
                        
         #       if c.name=="pareto_mp4":
         #           if c.pushed:
         #               print("pareto mp4")
        #    batch.draw()    
          #  self.quick_mat(query_df_list,x,y)
            #self.xy=self.quick_mat(query_df_list,x,y)  
        #    self.quick_mat(query_df_list,x,y)  
          #  self._graph_mat(self.saved_xy)   #x_vals,scaled_y_vals)
            #self.quick_mat(self.xy)                     
        return
   
    


    def create_query(self):
        #  take the global BUTTON_LIST and quesy sales_df to produce queries_slaes_df
   
        #print("create query doct in=",query_dict)
        stretched_query=[]
        query_list=[]
        for b in BUTTON_LIST:

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
                            
                            self.first_date_needed=False       
                            self.second_date_needed=True
                            
                            for c in BUTTON_LIST:
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
  
                            self.first_date_needed=False 
                            self.second_date_needed=False
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
    

                     #   print("streched query=",stretched_query) 
                        
                        
                        
    #   query_list=[query_list]
        if not self.second_date_needed:  #len(query_list)>0:     
            query_dict={"query":query_list} 
          #  print("query_dict before=",query_dict)
        
     #   if query_dict!=self.old_query_dict:
     #       self.old_query_dict=query_dict.copy()
        
            query_df_list=self.queries(query_dict)
         #   print(query_df_list)
            return query_dict,query_df_list
        else:
            return {},[]
            
       #     self.display_number_of_records(sales_df,query_df_list)
#        print("create query query dict out=",query_dict)  


       
        
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
                        try:
                           end=q[2]
                        except IndexError:
                           end=q[1] 
                        new_df=new_df[(pd.to_datetime(new_df[q[0]])>=pd.to_datetime(q[1])) & (pd.to_datetime(new_df[q[0]])<=pd.to_datetime(q[2]))].copy() 
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
    
  
      
    def _build_a_query(self,query_spec):
     #   print("build an entry query_name",query_name)
        print("Working on query:",query_spec)
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
     #   self.save(q_df,dd2.dash2_dict['sales']['save_dir'],query_name[0])   
        return new_df
    
           
    
    def queries(self,query_dict):
      #  self.query=sales_query_class()
      
     #   query_df=qdf.copy()
     #   dd2.dash2_dict['sales']['query_df']=query_df.copy()
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
            return []
    
   
    
    def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
    


    def plot_mat(self,mat_df,query_name):
      new_df=mat_df.groupby(mat_df.index,sort=True).sum()
      new_df['mat']=new_df['salesval'].rolling(365,axis=0).sum().round(0)

      styles1 = ['b-']
    # styles1 = ['bs-','ro:','y^-']
      linewidths = 1  # [2, 1, 4]
             
      fig, ax = pyplot.subplots()   #(figsize=(6,4))
      fig.patch.set_facecolor('None')
      fig.autofmt_xdate()
      ax=new_df.iloc[365:][['mat']].plot(use_index=True,grid=True,fontsize=6,style=styles1, lw=linewidths,legend=False)
      ax.patch.set_facecolor('None')
      ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places
    #  ax.xaxis.set_major_formatter(StrMethodFormatter('{yyyy/mmm}')) # 2 decimal places

      ax.set_title(query_name[:105]+"\n"+query_name[105:205]+"\n"+query_name[205:300]+" $ sales MAT",fontsize= 6,color='white')
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
      self._save_fig("mat_chart","./",transparent=True,facecolor=fig.get_facecolor())  #self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
      self._save_fig("mat_chart_normal","./",transparent=False,facecolor='black') 
      plt.close()
      
      return   
 
      


    def plot_pareto_customer(self,par_df,query_name):
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
            
            self._save_fig("pareto_customer_chart","./",transparent=True,facecolor=fig.get_facecolor())  #self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
            self._save_fig("pareto_customer_chart_normal","./",transparent=False,facecolor='black') 

        #    self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_customer_$_ranking",output_dir)
            plt.close()
        else:
            print("pareto customer nothing plotted. no records for ",par_df.shape)
 
        return




    def plot_pareto_product(self,par2_df,query_name):
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
            
            self._save_fig("pareto_product_chart","./",transparent=True,facecolor=fig.get_facecolor())  #self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
            self._save_fig("pareto_product_chart_normal","./",transparent=False,facecolor='black') 

        #    self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_customer_$_ranking",output_dir)
            plt.close()
        else:
            print("pareto product nothing plotted. no records for ",par_df.shape)
 
        return










def position_text_in_active_window(text,*,x,y):
    batch = pyglet.graphics.Batch()
   # canvas={}
   # canvas[1] = pyglet.text.Label(text, x=x, y=y, batch=batch)
    pyglet.text.Label(text, x=x, y=y, batch=batch)
    batch.draw()
   # return batch








#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

config = pyglet.gl.Config(double_buffer=True)      
window = QueryWindow(1600,900,resizable=False,caption="Query on "+str(load_dir)+str(load_file),config=config,visible=True)
keys = key.KeyStateHandler()
window.push_handlers(keys)
#canvas={}
#batch = pyglet.graphics.Batch()
#position_text_in_active_window("LOADING - Sales Transactions",x=0,y=0)

os.chdir("/home/tonedogga/Documents/python_dev")
#print("\nLoading sales_df.pkl.....")
st=sales_trans()
sales_df=st.load_pickle(load_dir,load_file)

query_df_list=[]
global query_dict
query_dict={}
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# define and make live areas on window as pushable buttons

class query_window_object(object):
    def __init__():
        pass



class button_object(query_window_object):
    def __init__(self,*,name,sub_name,x_start,x_len,y_start,y_len,colour,pushed_colour,title,sub_title,active,visible,movable,pushable,floating,pushed,contains_list,unique_list,allow_multiple_selections):
  #      super().__init__()
        self.name=name
        self.sub_name=sub_name
        self.x_start=x_start
        self.x_len=x_len
        self.y_start=y_start
        self.y_len=y_len
        self.colour=colour
        self.inactive_colour=(40,50,60)
        self.pushed_colour=pushed_colour
        self.button_type=0
        self.title=title
        self.sub_title=sub_title
        self.active=active
        self.visible=visible
        self.movable=movable
        self.pushable=pushable
        self.floating=floating
        self.mouse_over=False
        self.pushed=pushed
        self.contains_list=contains_list
        self.unique_list=unique_list
        self.unique_list_start_point=1
        self.allow_multiple_selections=allow_multiple_selections
        self.date_query=[]
        self.unique_list_display_length=30
        self.selected_value_list=[""]
       # self.button_array=button_array
       # self.button_df=button_df.copy()
        
 
    
 
# class list_of_buttons(query_window_object):
#       def __init__(self,*,list_name,list_of_buttons):
#         super().__init__(self)
#         self.list_name=list_name
#         self.list_of_buttons=list_of_buttons
        
  
 
    
class buttons(object):
    def setup_buttons(self,filename,sales_df_dict,unique_dict):
         bdf=pd.read_csv(filename,index_col=False,header=0)
        
         for index, row in bdf.iterrows():
             colour=(int(row['colour1']),int(row['colour2']),int(row['colour3']))
             pushed_colour=(int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
             button=button_object(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),pushable=bool(row['pushable']),floating=False,pushed=bool(row['pushed']),contains_list=bool(row['contains_list']),unique_list=[" "],allow_multiple_selections=True)    
             BUTTON_LIST.append(button)
             
         x_start=20   
         y_start=650
         
     #    print("sales_df_dict=",sales_df_dict)
    #     i=0
         for fields,values in sales_df_dict.items():
             if values==1:
                 colour=(200,200,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields),sub_name="AND",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title=str(fields),sub_title="AND",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
                 button.button_type=1   
                 BUTTON_LIST.append(button)
       
                 x_start+=40
                 colour=(200,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields),sub_name="OR",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="OR",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
                 button.button_type=1   
                 BUTTON_LIST.append(button)
                 
                 x_start+=40
                 colour=(0,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields),sub_name="NOT",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="NOT",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields],allow_multiple_selections=True)    
                 button.button_type=1   
                 BUTTON_LIST.append(button)
                            
                 x_start+=90
  
             elif values==2:   # dfates

                 colour=(255,0,0)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields)+"_start",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="START",active=True,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields+"_start"],allow_multiple_selections=False)    
                 button.button_type=1   
                 BUTTON_LIST.append(button)
                 x_start+=90
 
                 colour=(255,0,0)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields)+"_end",sub_name="BD",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="date",sub_title="END",active=False,visible=True,movable=False,pushable=False,floating=False,pushed=False,contains_list=True,unique_list=unique_dict[fields+"_end"],allow_multiple_selections=False)    
                 button.button_type=1   
                 BUTTON_LIST.append(button)
                 
   
 
    
                 x_start+=90
            #     if i%2:   # st.display_number_of_records(sales_df,query_dict,query_df_list)
            #        y_start+=10
            #     else:   
            #        y_start-=10

            #     i+=1
                 
         self._resize_buttons(window.x_max,window.y_max)
    
         return BUTTON_LIST
    
    
    
    def _resize_buttons(self,x_max,y_max):
       # batch = pyglet.graphics.Batch()
        for button in BUTTON_LIST: 
            if button.x_start<0:
               button.x_start=0
            if button.x_start>x_max:
               button.x_start=x_max
            if button.y_start<0:
               button.y_start=0
            if button.y_start>y_max:
               button.y_start=y_max
        
        
            if button.x_len<0:
               button.x_len=0
            if button.x_len>x_max:
               button.x_len=x_max
            if button.y_len<0:
               button.y_len=0
            if button.y_len>y_max:
               button.y_len=y_max
        
        
            if button.x_start+button.x_len>x_max:
               button.x_len=x_max-button.x_start
            
            if button.y_start+button.y_len>y_max:
               button.y_len=y_max-button.y_start

           
           
        
    
#-------------------------------------------------------------------------------------------------------------------------


def _index_containing_substring(the_list, substring,start):
     for i, s in enumerate(the_list):
         if substring in s[0]:
             return i
     return start
     
 
def _shortcut(char,x,y):  
     for b in BUTTON_LIST:
         b.mouse_over=False
         if ((not b.floating) & b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
             b.mouse_over=True

 #        if ((not b.floating) & b.active & b.visible & b.mouse_over):
             b.unique_list_start_point=_index_containing_substring(b.unique_list,char,b.unique_list_start_point)
             move_and_draw_pointer(x,y)
  #           return True
     #return False   
  #    return b.selected_value_list[b.unique_list_start_point]
    
  
def _list_move(value,x,y): 
     for b in BUTTON_LIST:
         b.mouse_over=False
         if ((not b.floating) & b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
             b.mouse_over=True

         
     #    if ((not b.floating) & b.active & b.visible & b.mouse_over):
             if (b.unique_list_start_point>=1) & (b.unique_list_start_point<=len(b.unique_list)-1):
                 b.unique_list_start_point+=value
                 if b.unique_list_start_point<1:
                     b.unique_list_start_point=1
                 if b.unique_list_start_point>=len(b.unique_list)-1:
                     b.unique_list_start_point=len(b.unique_list)-1
                 move_and_draw_pointer(x,y)
    
 
    
 


def act_on_key_state(x,y,symbol,modifiers):
    #keys = key.KeyStateHandler()
    #window.push_handlers(keys)
    # Check if the spacebar is currently pressed:
    if keys[key.F1]:  
        for b in BUTTON_LIST:
           if ((not b.floating) & b.active & b.visible & b.mouse_over):
               if (not b.unique_list[b.unique_list_start_point] in b.selected_value_list) & (b.unique_list[b.unique_list_start_point]!=" "):
                   if b.allow_multiple_selections:
                       b.selected_value_list.append(b.unique_list[b.unique_list_start_point]) 
                       query_dict,query_df_list=st.create_query()
                       st.display_number_of_records(sales_df,query_dict,query_df_list,20,880) 
                   else:
                       if len(b.selected_value_list)<=2:
                           if len(b.selected_value_list)<=1:
                               b.selected_value_list=[" "] 
                           elif len(b.selected_value_list)==2:
                               del b.selected_value_list[-1]   #remove(b.unique_list[b.unique_list_start_point]) 
                           b.selected_value_list.append(b.unique_list[b.unique_list_start_point]) 
                           query_dict,query_df_list=st.create_query()
                           st.display_number_of_records(sales_df,query_dict,query_df_list,20,880) 
   
                       else:
                           print("multiple selections not allowed on this field")
              #     move_and_draw_pointer(x,y)
                   
    elif keys[key.F2]:
        for b in BUTTON_LIST:
             if ((not b.floating) & b.active & b.visible & b.mouse_over):
                if b.unique_list[b.unique_list_start_point] in b.selected_value_list:               
                   if len(b.selected_value_list)>=2:                       
                        b.selected_value_list.remove(b.unique_list[b.unique_list_start_point]) 
                        query_dict,query_df_list=st.create_query()
                        st.display_number_of_records(sales_df,query_dict,query_df_list,20,880) 
                        if b.name=='date_end':
                            b.active=False

                   else:
                        print("selections empty")
               #     move_and_draw_pointer(x,y)
                    
    elif keys[key.UP]:
          _list_move(-1,x,y)
    elif keys[key.DOWN]:
          _list_move(1,x,y)
    elif keys[key.PAGEUP]:
          _list_move(-window.list_page_move_length,x,y)
    elif keys[key.PAGEDOWN]:
          _list_move(window.list_page_move_length,x,y)  
    elif keys[key.ESCAPE]:
         window.close()
         sys.exit()
    else:      
         _shortcut(pyglet.window.key.symbol_string(symbol),x,y)

    return 


#--------------------------------------------------------------------------------------------------------------------





def check_for_mouse_over(x,y):
  #  print("button object check for collissions",x,y)
    # if x,y is over any button display on screen
    # button list is global
#    over_button=[]

    batch = pyglet.graphics.Batch()
    for b in BUTTON_LIST:
        #b.active
        if (b.visible & b.active & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
            if MY_DEBUG: 
                position_text_in_active_window(b.name+"\nActive="+str(b.active)+"\nVisible="+str(b.visible)+" Pushed="+str(b.pushed)+" Movable="+str(b.movable)+" Floating="+str(b.floating),x=x,y=y)
            position_list_in_active_window(x=x-10,y=y-65,input_list=b.unique_list[b.unique_list_start_point:b.unique_list_start_point+b.unique_list_display_length],selection_list_input=True)
            position_list_in_active_window(x=x-10,y=y+len(b.selected_value_list)*20+10,input_list=b.selected_value_list,selection_list_input=False)
        
      
    batch.draw() 


        
def draw_buttons():
    batch = pyglet.graphics.Batch()
    for b in BUTTON_LIST:
        if b.visible:
            batch=_draw_button(b,batch)
       # if b.visible & b.active:    
       #     position_list_in_active_window(x=b.x_start,y=b.y_start+40,input_list=b.selected_value_list)
    batch.draw()        
    #return batch    



        
           
       
       
def _draw_button(button,batch):       
    if button.active:
        if button.contains_list:
            position_text_in_active_window(str(len(button.selected_value_list)-1),x=button.x_start,y=button.y_start+20)
        position_text_in_active_window(button.title,x=button.x_start,y=button.y_start-20)
        position_text_in_active_window(button.sub_title,x=button.x_start,y=button.y_start-35)        
        if button.button_type==1:
            _draw_rect(button.x_start,button.y_start-60,button.x_len+10,button.y_len,colour=(255,0,0),batch=batch)
            
        if not button.pushed:
            _draw_solid_rect(button.x_start,button.y_start,button.x_len,button.y_len,colour=button.colour,batch=batch)
        else:    
          #  batch=_draw_rect(button.x_start,button.x_len,button.y_start,button.y_len,colour=button.pushed_colour,batch=batch)
            _draw_solid_rect(button.x_start,button.y_start,button.x_len,button.y_len,colour=button.pushed_colour,batch=batch)
 
    else:
       # batch=_draw_rect(button.x_start,button.x_len,button.y_start,button.y_len,colour=button.inactive_colour,batch=batch)       
        _draw_solid_rect(button.x_start,button.y_start,button.x_len,button.y_len,colour=button.inactive_colour,batch=batch)       
    return batch     
    
 
    
 
def _draw_rect(x,y,x_len,y_len,colour,batch):
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
    
  #  batch.add(4, GL_QUADS, None, 'v2f', 't2f')
   # batch.add(4, pyglet.gl.GL_RECTS, None,
   #                          ('v2i', (x, y, x+x_len, y)),             
   #                          ('t2i', (x,y+y_len, x+x_len,y+y_len))
                             
 #   )
    # batch.add(2, pyglet.gl.GL_LINES, None,
    #                          ('v2i', (x+x_len, y, x+x_len, y+y_len)),             
    #                          ('c3B', final_colour)
    # )
    # batch.add(2, pyglet.gl.GL_LINES, None,
    #                          ('v2i', (x+x_len, y+y_len, x, y+y_len)),             
    #                          ('c3B', final_colour)
    # )
    # batch.add(2, pyglet.gl.GL_LINES, None,
    #                          ('v2i', (x, y+y_len, x, y)),             
    #                          ('c3B', final_colour)
   # )
    batch.draw()
  #  return batch
    
    
 
    
    
 
def _draw_solid_rect(x,y,x_len,y_len,colour,batch):
  #  final_colour=colour+colour
    
    
    rectangle = shapes.Rectangle(x, y, x_len, y_len, color=colour, batch=batch)
    rectangle.opacity = 128
    rectangle.rotation = 0
    
    batch.draw()
#    # print("final colour=",final_colour)
       
    
# # circle = shapes.Circle(700, 150, 100, color=(50, 225, 30), batch=batch)
# square = shapes.Rectangle(200, 200, 200, 200, color=(55, 55, 255), batch=batch)
# rectangle = shapes.Rectangle(250, 300, 400, 200, color=(255, 22, 20), batch=batch)
# rectangle.opacity = 128
# rectangle.rotation = 33
# line = shapes.Line(100, 100, 100, 200, width=19, batch=batch)
# line2 = shapes.Line(150, 150, 444, 111, width=4, color=(200, 20, 20), batch=batch)
    
 
    
    
def draw_pointers(x,y,x_max,y_max):
    batch = pyglet.graphics.Batch()
 #   batch.add(2, pyglet.gl.GL_LINES, None,
  #                           ('v2i', (0, 0, x, y,0,y_max)),
                         #    ('v2i', (x_max,0, x, y,x_max,y_max)), 
  #                           ('c3B', (255, 0, 0, 255, 255, 255))
  #  )
    batch.add(2, pyglet.gl.GL_LINES, None,
                              ('v2i', (0, 0, x, y)),             
                              ('c3B', (255, 0, 0, 255, 255, 255))
    )

    batch.add(2, pyglet.gl.GL_LINES, None,
                              ('v2i', (0, y_max, x, y)),             
                              ('c3B', (0, 255, 0, 255, 255, 255))
    )
    
    batch.add(2, pyglet.gl.GL_LINES, None,
                              ('v2i', (x_max,0, x, y)),             
                              ('c3B', (0, 0, 255, 255, 255, 255))
    )

    batch.add(2, pyglet.gl.GL_LINES, None,
                              ('v2i', (x_max, y_max, x, y)),             
                              ('c3B', (60, 70, 20, 255, 255, 255))
    )
    batch.draw() 
   # return batch 
    
    
    
   
   
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   


def position_list_in_active_window(*,x,y,input_list,selection_list_input):
    batch = pyglet.graphics.Batch()
    for elem in input_list:
        if selection_list_input and (elem=="" or elem==" "):       
            elem="|__EMPTY__|"
     
        pyglet.text.Label(elem, x=x, y=y,batch=batch)
        y-=16
    batch.draw()



def _display_text_in_active_window(text):
    batch = pyglet.graphics.Batch()
   # canvas={}
   # canvas[1] = pyglet.text.Label(text, x=x, y=y, batch=batch)
    pyglet.text.Label(text, x=5, y=window.get_size()[1]-12, batch=batch)
  #  window.clear()
    batch.draw()
    



def move_and_draw_pointer(x,y):
 
   #     batch = pyglet.graphics.Batch()
        window.clear()
      #  draw_buttons()
     #   st.display_number_of_records(sales_df,query_dict,query_df_list)
        if MY_DEBUG:
            draw_pointers(x,y,window.x_max,window.y_max)
        check_for_mouse_over(x,y)
     #   position_text_in_active_window("Queried Sales Transactions size="+str(st.qdf_size)+" rows",x=100,y=970)
        st.display_number_of_records(sales_df,query_dict,query_df_list,20,880)
#          position_list_in_active_window(x=x,y=y,input_list=b.unique_list[b.unique_list_start_point:])

        clock.tick()
        if MY_DEBUG:
            position_text_in_active_window("fps="+str(int(clock.get_fps()))+" size="+str(window.get_size())+" loc="+str(window.get_location())+" Pos=("+str(x)+","+str(y)+") dx=("+str(dx)+","+str(dy)+")",x=0,y=5)
 
    
 
def show_buttons(buttons):
    for b in buttons:
        print(b.name,b.sub_name,b.title,b.sub_title,b.second_date_needed)
 
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def main():

   # print(sales_df.info(),sales_df.shape)
    unique_dict=st.find_uniques(sales_df)
  #  print("unuqie dict",unique_dict)
   
  
    b=buttons()
    BUTTON_LIST=b.setup_buttons('./dash2/buttons.csv',st.sales_df_dict,unique_dict)
 #   show_buttons(BUTTON_LIST)
    pyglet.app.run()
    window.close()


main()


    #x, y = window.get_location()
    #window.set_location(x + 20, y + 20)
    
    
    #window = MyWindow(1200,1200,resizable=True,caption="Queries",visible=True)
    
    #pyglet.clock.schedule_interval(draw_batch, 1.0/60.0)
    #window.switch_to()
    # signify that one frame has passed
    #pyglet.clock.tick()
    # poll the operating system event queue
    #window.dispatch_events()
    
    # getting window size 
    #value = window.get_size() 
    #window.activate() 
    

 
    
    