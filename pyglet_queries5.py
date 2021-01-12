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

import pyglet
from pyglet import clock
from pyglet import gl
from pyglet.gl import *

from pyglet.window import key
from pyglet.window import mouse

from pyglet import shapes



#from time import time

MY_DEBUG=False #True   #False
BUTTON_LIST=[]
#BUTTON_COLOR=(200,100,200)
#global batch


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
      #  batch=check_for_collisions(x,y)
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
                if ((not b.floating) & b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
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
                _list_move(scroll_y,x,y)
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
   #     check_for_collisions(x,y)
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
        self.sales_df_dict={
            "cat":True,
            "code":True,
            "costval":False,
            "doctype":False,
            "docentryno":False,
            "linenumber":False,
            "location":True,
            "product":True,
            "productgroup":True,
            "qty":False,
            "refer":False,
            "salesrep":True,
            "saleval":False,
            "territory":False,
            "date":False,
            "glset":True,
            "specialpricecat":True,
            "period":False}
        
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
        
    
    def load_pickle(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            return pd.read_pickle(save_dir+savefile)
        else:
            print("load sales_df error.")
            return
        
 

    def find_uniques(self,sales_df):
        unique_dict={}
        for k,v in self.sales_df_dict.items():
            if v:
                if k=="specialpricecat":
                    unique_dict[k]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
                 #   unique_dict[k]=[int(i) for i in unique_dict[k]]
                else:    
                    unique_dict[k]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
                

 
        return unique_dict



    def display_number_of_records(self,sales_df,query_dict,query_df_list):
        position_text_in_active_window("Sales Transactions size="+str(sales_df.shape[0])+" rows",x=self.display_x,y=self.display_y)
        self.total_sales=int(sales_df['salesval'].sum())
        self.total_qty=int(sales_df['qty'].sum())
        position_text_in_active_window("Sales Transactions $"+str(self.total_sales),x=self.display_x,y=self.display_y-15)
        position_text_in_active_window("Sales Qty="+str(self.total_qty),x=self.display_x,y=self.display_y-30)
        
      #  if len(query_df_list)>0:
        #    print("query df list=\n",query_df_list)   
        if len(query_df_list)==1:
            qdf=query_df_list[0]
            if self.qdf_size!=qdf.shape[0]: 
                self.qdf_size=qdf.shape[0]
                print(self.qdf_size) 
                self.qdf_sales=int(qdf['salesval'].sum())
                self.qdf_qty=int(qdf['qty'].sum())
                self.display=True
        #    else:
        #        self.display=False
        if self.display:
            if query_dict:
                self.query_dict_save=query_dict
            position_text_in_active_window("Query="+str(self.query_dict_save),x=self.display_x,y=self.display_y-55)
            position_text_in_active_window("Queried Sales Transactions size="+str(self.qdf_size)+" rows",x=self.display_x,y=self.display_y-70)
            position_text_in_active_window("Query Transactions $"+str(self.qdf_sales),x=self.display_x,y=self.display_y-85)
            position_text_in_active_window("Query Qty="+str(self.qdf_qty),x=self.display_x,y=self.display_y-100)
                                             
               


    def create_query(self):
        #  take the global BUTTON_LIST and quesy sales_df to produce queries_slaes_df
   
        #print("create query doct in=",query_dict)
        query_list=[]
        for b in BUTTON_LIST:
            qlist=b.selected_value_list[1:]
            if len(qlist)>0:
                 #   print("create query",b.name)
                    if b.name=="specialpricecat":
 #                       stretched_query=[b.sub_name]+[(b.name,int(i)) if isinstance(i, numbers.Number) else (b.name,i) for i in qlist]
                        stretched_query=[b.sub_name]+[(b.name,float(i)) if type(i) == int or float else (b.name,i) for i in qlist]
 
                        print("sq=",stretched_query) 
                    else:                       
                        stretched_query=[b.sub_name]+[(b.name,i) for i in qlist]

                    query_list.append(stretched_query)
    

             #   query_list=[query_list]
        if True:  #len(query_list)>0:     
            query_dict={"query":query_list} 
          #  print("query_dict before=",query_dict)
        
     #   if query_dict!=self.old_query_dict:
     #       self.old_query_dict=query_dict.copy()
        
            query_df_list=self.queries(query_dict)
       #     self.display_number_of_records(sales_df,query_df_list)
        print("create query query dict out=",query_dict)  
        return query_dict,query_df_list

       
        
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
    
  
      
    def _build_a_query(self,query_spec):
     #   print("build an entry query_name",query_name)
        print("query spec=",query_spec)
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
    
    








#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

config = pyglet.gl.Config(double_buffer=True)      
window = QueryWindow(1500,1200,resizable=False,caption="Salestrans Queries",config=config,visible=True)
keys = key.KeyStateHandler()
window.push_handlers(keys)
#canvas={}
#batch = pyglet.graphics.Batch()
os.chdir("/home/tonedogga/Documents/python_dev")
st=sales_trans()
sales_df=st.load_pickle("./dash2_saves/","raw_savefile.pkl")
query_df_list=[]
global query_dict
query_dict={}
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# define and make live areas on window as pushable buttons

class query_window_object(object):
    def __init__():
        pass



class button_object(query_window_object):
    def __init__(self,*,name,sub_name,x_start,x_len,y_start,y_len,colour,pushed_colour,title,sub_title,active,visible,movable,floating,pushed,toggle,unique_list):
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
        self.floating=floating
        self.mouse_over=False
        self.pushed=pushed
        self.toggle=toggle
        self.unique_list=unique_list
        self.unique_list_start_point=1
        
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
             button=button_object(name=str(row['name']),sub_name="",x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,pushed_colour=pushed_colour,title=str(row['title']),sub_title=str(""),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),floating=False,pushed=bool(row['pushed']),toggle=bool(row['toggle']),unique_list=[" "])    
             BUTTON_LIST.append(button)
             
         x_start=30   
         y_start=700
    #     i=0
         for fields,values in sales_df_dict.items():
             if values:
                 colour=(200,200,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields),sub_name="AND",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title=str(fields),sub_title="AND",active=True,visible=True,movable=False,floating=False,pushed=False,toggle=True,unique_list=unique_dict[fields])    
                 button.button_type=1   
                 BUTTON_LIST.append(button)
       
                 x_start+=40
                 colour=(200,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields),sub_name="OR",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="OR",active=True,visible=True,movable=False,floating=False,pushed=False,toggle=True,unique_list=unique_dict[fields])    
                 button.button_type=1   
                 BUTTON_LIST.append(button)
                 
                 x_start+=40
                 colour=(0,0,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields),sub_name="NOT",x_start=x_start,x_len=27,y_start=y_start+20,y_len=15,colour=colour,pushed_colour=pushed_colour,title="",sub_title="NOT",active=True,visible=True,movable=False,floating=False,pushed=False,toggle=True,unique_list=unique_dict[fields])    
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
                   b.selected_value_list.append(b.unique_list[b.unique_list_start_point]) 
                   query_dict,query_df_list=st.create_query()
                   st.display_number_of_records(sales_df,query_dict,query_df_list) 
              #     move_and_draw_pointer(x,y)
                   
    elif keys[key.F2]:
        for b in BUTTON_LIST:
             if ((not b.floating) & b.active & b.visible & b.mouse_over):
                if b.unique_list[b.unique_list_start_point] in b.selected_value_list:
                    b.selected_value_list.remove(b.unique_list[b.unique_list_start_point]) 
                    query_dict,query_df_list=st.create_query()
                    st.display_number_of_records(sales_df,query_dict,query_df_list) 
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





def check_for_collisions(x,y):
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
            position_list_in_active_window(x=x,y=y-65,input_list=b.unique_list[b.unique_list_start_point:b.unique_list_start_point+b.unique_list_display_length])
            position_list_in_active_window(x=x,y=y+len(b.selected_value_list)*20,input_list=b.selected_value_list)
        
      
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


def position_list_in_active_window(*,x,y,input_list):
    batch = pyglet.graphics.Batch()
    for elem in input_list:
        pyglet.text.Label(elem, x=x, y=y,batch=batch)
        y-=16
    batch.draw()



def position_text_in_active_window(text,*,x,y):
    batch = pyglet.graphics.Batch()
   # canvas={}
   # canvas[1] = pyglet.text.Label(text, x=x, y=y, batch=batch)
    pyglet.text.Label(text, x=x, y=y, batch=batch)
    batch.draw()
   # return batch



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
        check_for_collisions(x,y)
     #   position_text_in_active_window("Queried Sales Transactions size="+str(st.qdf_size)+" rows",x=100,y=970)
        st.display_number_of_records(sales_df,query_dict,query_df_list)
#          position_list_in_active_window(x=x,y=y,input_list=b.unique_list[b.unique_list_start_point:])

        clock.tick()
        if MY_DEBUG:
            position_text_in_active_window("fps="+str(int(clock.get_fps()))+" size="+str(window.get_size())+" loc="+str(window.get_location())+" Pos=("+str(x)+","+str(y)+") dx=("+str(dx)+","+str(dy)+")",x=0,y=5)
 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def main():

   # print(sales_df.info(),sales_df.shape)
    unique_dict=st.find_uniques(sales_df)
  #  print("unuqie dict",unique_dict)
   
  
    b=buttons()
    BUTTON_LIST=b.setup_buttons('./dash2/buttons.csv',st.sales_df_dict,unique_dict)

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
    

 
    
    