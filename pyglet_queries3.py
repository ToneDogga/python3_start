#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:10:04 2020

@author: tonedogga
"""

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

MY_DEBUG=False   #True   #False
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
        
        #print(self.get_size())
        
        # get window location
        #x, y = window.get_location()
        #window.set_location(x + 20, y + 20) 
 
    # button_list is global    
 
    #  buttons=get_button_details()
        
        
        
    def on_key_press(self, symbol, modifiers):
        if symbol == key.A:
            if MY_DEBUG:
           # print('The "A" key was pressed.')
                text='The "A" key was pressed.'
                for b in BUTTON_LIST:
                    if ((not b.floating) & b.active & b.visible & (self.x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                        b.selected_value_list.append(b.unique_list[b.unique_list_start_point]) 
                        move_and_draw_pointer(x,y,0,0)
                _display_text_in_active_window(text)
        #     self.window.set_visible()
        elif symbol == key.LEFT:
            if MY_DEBUG:
                text='The left arrow key was pressed.'
#            print('The left arrow key was pressed.')
                _display_text_in_active_window(text)
        elif symbol == key.ENTER:
            if MY_DEBUG:
                text='The enter key was pressed.'
        #    print('The enter key was pressed.')
                _display_text_in_active_window(text)

        
    
    
    def on_key_release(self, symbol, modifiers):
        
        pass
    
    
    def on_mouse_enter(self,x, y):
        pass

    def on_mouse_leave(self,x, y):
        pass
    
    def on_mouse_motion(self,x, y, dx, dy):
      #  fps_display.draw()
      #  batch=check_for_collisions(x,y)
        move_and_draw_pointer(x,y,dx,dy)
       
   
 
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
                    move_and_draw_pointer(x,y,dx,dy)
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
          #  canvas={}
          #  print('The left mouse button was pressed. x=',x,"y=",y)
   #         batch=_check_for_collisions(x,y,batch)
           # if MY_DEBUG:
           #     text="the left mouse button was pressed. x="+str(x)+" y="+str(y)
           #     _display_text_in_active_window(text)
            for b in BUTTON_LIST:
                if ((not b.floating) & b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                    b.pushed=not b.pushed
            #batch=display_buttons(batch)       

        elif button == mouse.RIGHT:
           # print('The right mouse button was pressed.')
            if MY_DEBUG:
                text="the right mouse button was pressed. x="+str(x)+" y="+str(y)

                _display_text_in_active_window(text)
    
    
  
    
  
    
    def on_mouse_scroll(self,x, y, scroll_x, scroll_y):
        for b in BUTTON_LIST:
            if ((not b.floating) & b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                if (b.unique_list_start_point>=1) & (b.unique_list_start_point<=(b.unique_list_len-1)):
                    b.unique_list_start_point+=scroll_y
                    if b.unique_list_start_point<1:
                        b.unique_list_start_point=1
                    if b.unique_list_start_point>b.unique_list_len-1:
                        b.unique_list_start_point=b.unique_list_len-1
                  #  b.selected_value_list.append(b.unique_list[b.unique_list_start_point])    
                #    position_list_in_active_window(x=x,y=y,input_list=b.unique_list[b.unique_list_start_point:])
                move_and_draw_pointer(x,y,0,0)
        if MY_DEBUG:
            text="mouse scroll x,y,scroll_x,scroll_y"+str(x)+" "+str(y)+" "+str(scroll_x)+" "+str(scroll_y)
            _display_text_in_active_window(text)

        
   

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
                unique_dict[k]=[" "," "]+sorted(pd.Series(sales_df[k]).astype(str).unique())
                
        return unique_dict



















#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

config = pyglet.gl.Config(double_buffer=True)      
window = QueryWindow(1200,1200,resizable=False,caption="Salestrans Queries",config=config,visible=True)
#canvas={}
#batch = pyglet.graphics.Batch()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# define and make live areas on window as pushable buttons

class query_window_object(object):
    def __init__():
        pass



class button_object(query_window_object):
    def __init__(self,*,name,x_start,x_len,y_start,y_len,colour,pushed_colour,title,active,visible,movable,floating,pushed,toggle,unique_list):
  #      super().__init__()
        self.name=name
        self.x_start=x_start
        self.x_len=x_len
        self.y_start=y_start
        self.y_len=y_len
        self.colour=colour
        self.inactive_colour=(40,50,60)
        self.pushed_colour=pushed_colour
        self.button_type=0
        self.title=title
        self.active=active
        self.visible=visible
        self.movable=movable
        self.floating=floating
        self.pushed=pushed
        self.toggle=toggle
        self.unique_list=unique_list
        self.unique_list_start_point=1
        self.unique_list_len=len(self.unique_list)
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
             button=button_object(name=str(row['name']),x_start=int(row['x_start']),x_len=int(row['x_len']),y_start=int(row['y_start']),y_len=int(row['y_len']),colour=colour,pushed_colour=pushed_colour,title=str(row['title']),active=bool(row['active']),visible=bool(row['visible']),movable=bool(row['movable']),floating=False,pushed=bool(row['pushed']),toggle=bool(row['toggle']),unique_list=[])    
             BUTTON_LIST.append(button)
             
         x_start=0   
         y_start=990
         i=0
         for fields,values in sales_df_dict.items():
             if values:
                 colour=(200,200,200)   #int(row['colour1']),int(row['colour2']),int(row['colour3']))
                 pushed_colour=(100,100,100)  #int(row['pcolour1']),int(row['pcolour2']),int(row['pcolour3']))
                 button=button_object(name=str(fields),x_start=x_start,x_len=10,y_start=y_start,y_len=30,colour=colour,pushed_colour=pushed_colour,title=str(fields),active=True,visible=True,movable=False,floating=False,pushed=False,toggle=True,unique_list=unique_dict[fields])    
                 x_start+=70
                 if i%2:
                    y_start+=10
                 else:   
                    y_start-=10
                 button.button_type=1   
                 BUTTON_LIST.append(button)
                 i+=1
                 
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



def check_for_collisions(x,y):
  #  print("button object check for collissions",x,y)
    # if x,y is over any button display on screen
    # button list is global
#    over_button=[]
    batch = pyglet.graphics.Batch()
#    batch=p_map(_check_button,BUTTON_LIST)
    for b in BUTTON_LIST:
        #b.active
        if (b.visible & b.active & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
#            position_text_in_active_window(b.name+"\nActive="+str(b.active)+"\nVisible="+str(b.visible)+" Pushed="+str(b.pushed)+" list="+str(b.unique_list),x=x,y=y)
            if MY_DEBUG: 
                position_text_in_active_window(b.name+"\nActive="+str(b.active)+"\nVisible="+str(b.visible)+" Pushed="+str(b.pushed)+" Movable="+str(b.movable)+" Floating="+str(b.floating),x=x,y=y)
            position_list_in_active_window(x=x,y=y-30,input_list=b.unique_list[b.unique_list_start_point:b.unique_list_start_point+b.unique_list_display_length])
            position_list_in_active_window(x=x,y=y+len(b.selected_value_list)*20,input_list=b.selected_value_list)
        #      over_button.append(True)
      #  else:
      #      over_button.append(False)
        #       print("button cooll",b.name,x,y)
    batch.draw() 
  #  return batch    
  #  return any(over_button),batch    
    
#def _check_button(b):
#    if (b.visible & b.active & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
#         position_text_in_active_window(b.name+"\nActive="+str(b.active)+"\nVisible="+str(b.visible)+" Pushed="+str(b.pushed)+" len="+str(b.unique_list_len),x=x,y=y)
#    return batch


        
def draw_buttons():
    batch = pyglet.graphics.Batch()
    for b in BUTTON_LIST:
        if b.visible:
            batch=_draw_button(b,batch)
        if b.visible & b.active:    
            position_list_in_active_window(x=b.x_start,y=b.y_start+40,input_list=b.selected_value_list)
    batch.draw()        
    #return batch    



        
           
       
       
def _draw_button(button,batch):       
    if button.active:
        position_text_in_active_window(button.title,x=button.x_start+10,y=button.y_start+button.y_len-20)
        
        if button.button_type==1:
            _draw_rect(button.x_start,button.y_start-10,button.x_len+55,button.y_len-10,colour=(255,0,0),batch=batch)
            
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
    
 
    
    
# def draw_pointers(x,y):
#     batch = pyglet.graphics.Batch()
#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (0, 0, x, y)),             
#                              ('c3B', (255, 0, 0, 255, 255, 255))
#     )
    
#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (0, window.get_size()[1], x, y)),             
#                              ('c3B', (0, 255, 0, 255, 255, 255))
#     )
    
#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (window.get_size()[0],0, x, y)),             
#                              ('c3B', (0, 0, 255, 255, 255, 255))
#     )

#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (window.get_size()[0], window.get_size()[1], x, y)),             
#                              ('c3B', (60, 70, 20, 255, 255, 255))
#     )
#     batch.draw() 
#    # return batch 
    
    
    
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
    



def move_and_draw_pointer(x,y,dx,dy):
 
   #     batch = pyglet.graphics.Batch()
        window.clear()
      #  draw_buttons()
        if MY_DEBUG:
            draw_pointers(x,y,window.x_max,window.y_max)
        check_for_collisions(x,y)
#          position_list_in_active_window(x=x,y=y,input_list=b.unique_list[b.unique_list_start_point:])

        clock.tick()
        if MY_DEBUG:
            position_text_in_active_window("fps="+str(int(clock.get_fps()))+" size="+str(window.get_size())+" loc="+str(window.get_location())+" Pos=("+str(x)+","+str(y)+") dx=("+str(dx)+","+str(dy)+")",x=0,y=5)
    #    batch=draw_buttons_to_batch(x_max=window.x_max,y_max=window.y_max,batch=batch)
    #    window.clear()
     #   batch.draw() 
    #    for index in list(canvas):
    #        canvas[index].delete()
    #        del(canvas[index])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def main():
    os.chdir("/home/tonedogga/Documents/python_dev")
    st=sales_trans()
    sales_df=st.load_pickle("./dash2_saves/","raw_savefile.pkl")
   # print(sales_df.info(),sales_df.shape)
    unique_dict=st.find_uniques(sales_df)
  #  print("unuqie dict",unique_dict)
   
  
    b=buttons()
    BUTTON_LIST=b.setup_buttons('./dash2/buttons.csv',st.sales_df_dict,unique_dict)
   # print("button_list length=",len(BUTTON_LIST))
  # batch = pyglet.graphics.Batch()
  #  batch=display_buttons(batch)
  #  window.clear()
  #  batch.draw()

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
    

 
    
    