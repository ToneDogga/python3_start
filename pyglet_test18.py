#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:10:04 2020

@author: tonedogga
"""

import numpy as np
import pandas as pd

import pyglet
from pyglet import clock
from pyglet import gl
from pyglet.gl import *

from pyglet.window import key
from pyglet.window import mouse

#from time import time

MY_DEBUG=True   #False
BUTTON_LIST=[]
BUTTON_COLOR=(200,100,200)


#########################################################################################################################




class MyWindow(pyglet.window.Window):
    def __init__(self,*args,**kwargs):
     #   super(MyWindow,self).__init__(*args,**kwargs)
        super(MyWindow,self).__init__(*args,**kwargs)
 
        #set window size
        self.set_minimum_size(700,700)
        self.set_maximum_size(2048, 2048)
        
        # get window size
        
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
                display_text_in_active_window(text)
        #     self.window.set_visible()
        elif symbol == key.LEFT:
            if MY_DEBUG:
                text='The left arrow key was pressed.'
#            print('The left arrow key was pressed.')
                display_text_in_active_window(text)
        elif symbol == key.ENTER:
            if MY_DEBUG:
                text='The enter key was pressed.'
        #    print('The enter key was pressed.')
                display_text_in_active_window(text)

        
    
    
    def on_key_release(self, symbol, modifiers):
        pass
    
    
    def on_mouse_enter(self,x, y):
        pass

    def on_mouse_leave(self,x, y):
        pass
    
    def on_mouse_motion(self,x, y, dx, dy):
      #  fps_display.draw()
      #  batch=check_for_collisions(x,y)
        draw_batch(x,y,dx,dy)
    
    def on_mouse_press(self,x,y,button, modifiers):
        if button == mouse.LEFT:
          #  canvas={}
          #  print('The left mouse button was pressed. x=',x,"y=",y)
   #         batch=_check_for_collisions(x,y,batch)
           # if MY_DEBUG:
           #     text="the left mouse button was pressed. x="+str(x)+" y="+str(y)
           #     display_text_in_active_window(text)
            for b in BUTTON_LIST:
                if (b.active & b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
                    b.pushed=not b.pushed
            #batch=display_buttons(batch)       




        elif button == mouse.RIGHT:
           # print('The right mouse button was pressed.')
            if MY_DEBUG:
                text="the right mouse button was pressed. x="+str(x)+" y="+str(y)

                display_text_in_active_window(text)
        
   
 
    def on_mouse_release(self,x, y, button, modifiers):
        pass


    def on_mouse_drag(self,x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT:
            if MY_DEBUG:
                text="mouse drag left x,y,dx,dy"+str(x)+" "+str(y)+" "+str(dx)+" "+str(dy)
                display_text_in_active_window(text)
          # print(text)
        elif buttons & mouse.RIGHT:
            if MY_DEBUG:
                text="mouse drag right x,y,dx,dy"+str(x)+" "+str(y)+" "+str(dx)+" "+str(dy)
                display_text_in_active_window(text)
          # print(text)
  
        

    
    def on_mouse_scroll(self,x, y, scroll_x, scroll_y):
        if MY_DEBUG:
            text="mouse scroll x,y,scroll_x,scroll_y"+str(x)+" "+str(y)+" "+str(scroll_x)+" "+str(scroll_y)
            display_text_in_active_window(text)

        
   

    def on_draw(self):  
        #self.clear()
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
        pass

        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





config = pyglet.gl.Config(double_buffer=True)      
window = MyWindow(1200,1200,resizable=False,caption="Queries",config=config,visible=True)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# define and make live areas on window as pushable buttons

class query_window_object():
    def __init__():
        pass



class button_object(query_window_object):
    def __init__(self,*,name,x_start,x_len,y_start,y_len,colour,pushed_colour,title,active,visible,pushed,toggle,button_array,button_df):
  #      super().__init__()
        self.name=name
        self.x_start=x_start
        self.x_len=x_len
        self.y_start=y_start
        self.y_len=y_len
        self.colour=colour
        self.inactive_colour=(40,50,60)
        self.pushed_colour=pushed_colour
        self.title=title
        self.active=active
        self.visible=visible
        self.pushed=pushed
        self.toggle=toggle
        self.button_array=button_array
        self.button_df=button_df.copy()
    

# class button_object(query_window_object):
#     def __init__(self,*,name,x_start,x_len,y_start,y_len,colour,title,active,visible,pushed,toggle,button_array,button_df):
#         super().__init__(self)
#         self.name=name
#         self.x_start=x_start
#         self.x_len=x_len
#         self.y_start=y_start
#         self.y_len=y_len
#         self.colour=colour
#         self.inactive_colour=(40,50,60)
#         self.title=title
#         self.active=active
#         self.visible=visible
#         self.pushed=pushed
#         self.toggle=toggle
#         self.button_array=button_array
#         self.button_df=button_df.copy()
 
        
        
    
#-------------------------------------------------------------------------------------------------------------------------


def _setup_buttons():
    
     button=button_object(name="Test_name",x_start=100,x_len=100,y_start=100,y_len=300,colour=(255,255,6),pushed_colour=(255,7,34),title="trest_title",active=True,visible=True,pushed=False,toggle=True,button_array=np.array(np.zeros((4,4),dtype=float)),button_df=pd.DataFrame([]))    
     BUTTON_LIST.append(button)
     button=button_object(name="Test_name2",x_start=1000,x_len=1000,y_start=100,y_len=300,colour=(255,10,6),pushed_colour=(255,255,7),title="trest_title2",active=True,visible=True,pushed=False,toggle=True,button_array=np.array(np.zeros((4,4),dtype=float)),button_df=pd.DataFrame([]))    
     BUTTON_LIST.append(button)
     button=button_object(name="Test_name3",x_start=680,x_len=60,y_start=500,y_len=30,colour=(200,10,6),pushed_colour=(255,255,7),title="trest_title3",active=False,visible=True,pushed=False,toggle=True,button_array=np.array(np.zeros((4,4),dtype=float)),button_df=pd.DataFrame([]))    
     BUTTON_LIST.append(button)
     button=button_object(name="Test_name4",x_start=680,x_len=60,y_start=50,y_len=30,colour=(200,100,6),pushed_colour=(255,255,7),title="trest_title4",active=True,visible=False,pushed=False,toggle=True,button_array=np.array(np.zeros((4,4),dtype=float)),button_df=pd.DataFrame([]))    
     BUTTON_LIST.append(button)


     return BUTTON_LIST




def _check_for_collisions(x,y,batch):
  #  print("button object check for collissions",x,y)
    # if x,y is over any button display on screen
    # button list is global
    over_button=[]
    for b in BUTTON_LIST:
        #b.active
        if (b.visible & (x>=b.x_start) & (x<(b.x_start+b.x_len)) & (y>=b.y_start) & (y<(b.y_start+b.y_len))):
            batch=batch_text_in_active_window(b.name+"\nActive="+str(b.active)+"\nVisible="+str(b.visible)+" Pushed="+str(b.pushed),x=x,y=y,batch=batch)
            over_button.append(True)
        else:
            over_button.append(False)
        #       print("button cooll",b.name,x,y)
    return batch    
  #  return any(over_button),batch    
    



        
def display_buttons(batch):
    for b in BUTTON_LIST:
        if b.visible:
            batch=_display_button(b,batch)
    return batch    


def _display_button(button,batch):
   # batch = pyglet.graphics.Batch()
    x_max=window.get_size()[0]
    y_max=window.get_size()[1]
    
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
    if button.active & button.visible:
        batch=batch_text_in_active_window(button.title,x=button.x_start+10,y=button.y_start+button.y_len-20,batch=batch)
        if not button.pushed:
            batch=_draw_rect(button.x_start,button.x_len,button.y_start,button.y_len,colour=button.colour,batch=batch)
        else:    
            batch=_draw_rect(button.x_start,button.x_len,button.y_start,button.y_len,colour=button.pushed_colour,batch=batch)
    else:
        batch=_draw_rect(button.x_start,button.x_len,button.y_start,button.y_len,colour=button.inactive_colour,batch=batch)       
        
    return batch 
 
    
 
def _draw_rect(x,x_len,y,y_len,colour,batch):
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
    
# =============================================================================
#     
#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (x, y, x+x_len, y)),             
#                              ('c3B', (255, 0, 0, 255, 255, 255))
#     )
#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (x+x_len, y, x+x_len, y+y_len)),             
#                              ('c3B', (255, 0, 0, 255, 255, 255))
#     )
#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (x+x_len, y+y_len, x, y+y_len)),             
#                              ('c3B', (255, 0, 0, 255, 255, 255))
#     )
#     batch.add(2, pyglet.gl.GL_LINES, None,
#                              ('v2i', (x, y+y_len, x, y)),             
#                              ('c3B', (255, 0, 0, 255, 255, 255))
#     )
# 
# =============================================================================
    
    
    return batch
 
    
 
    
 
    
 
    
#    window.clear()
#    clock.tick()
#    if MY_DEBUG:
#        batch_text_in_active_window("fps="+str(int(clock.get_fps()))+" size="+str(window.get_size())+" loc="+str(window.get_location())+" Pos=("+str(x)+","+str(y)+") dx=("+str(dx)+","+str(dy)+")",x=0,y=5,batch=batch)
   
  #  batch.draw() 
     

    
    
def _display_pointers(x,y,batch):
    batch.add(2, pyglet.gl.GL_LINES, None,
                             ('v2i', (0, 0, x, y)),             
                             ('c3B', (255, 0, 0, 255, 255, 255))
    )
    
    batch.add(2, pyglet.gl.GL_LINES, None,
                             ('v2i', (0, window.get_size()[1], x, y)),             
                             ('c3B', (0, 255, 0, 255, 255, 255))
    )
    
    batch.add(2, pyglet.gl.GL_LINES, None,
                             ('v2i', (window.get_size()[0],0, x, y)),             
                             ('c3B', (0, 0, 255, 255, 255, 255))
    )

    vertex_list=batch.add(2, pyglet.gl.GL_LINES, None,
                             ('v2i', (window.get_size()[0], window.get_size()[1], x, y)),             
                             ('c3B', (60, 70, 20, 255, 255, 255))
    )
    return batch    
    
    
    
   
   
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   


def draw_batch(x,y,dx,dy):
        window.clear()
        batch = pyglet.graphics.Batch()
        batch=_display_pointers(x,y,batch)
        batch=_check_for_collisions(x,y,batch)
        clock.tick()
        if MY_DEBUG:
            batch_text_in_active_window("fps="+str(int(clock.get_fps()))+" size="+str(window.get_size())+" loc="+str(window.get_location())+" Pos=("+str(x)+","+str(y)+") dx=("+str(dx)+","+str(dy)+")",x=0,y=5,batch=batch)
        batch=display_buttons(batch)
        batch.draw() 


    

def batch_text_in_active_window(text,*,x,y,batch):
  #  batch = pyglet.graphics.Batch()
   # canvas={}
   # canvas[1] = pyglet.text.Label(text, x=x, y=y, batch=batch)
    pyglet.text.Label(text, x=x, y=y, batch=batch)
    return batch



def display_text_in_active_window(text):
    batch = pyglet.graphics.Batch()
   # canvas={}
   # canvas[1] = pyglet.text.Label(text, x=x, y=y, batch=batch)
    pyglet.text.Label(text, x=5, y=window.get_size()[1]-12, batch=batch)
    window.clear()
    batch.draw()





#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def main():
    
    BUTTON_LIST=_setup_buttons()
 #   print("buttion_list=",BUTTON_LIST)
  #  batch = pyglet.graphics.Batch()
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
    

 
    
    