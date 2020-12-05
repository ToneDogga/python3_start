#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:10:04 2020

@author: tonedogga
"""

import pyglet
from pyglet.window import key
from pyglet.window import mouse

class MyWindow(pyglet.window.Window):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        #set window size
        self.set_minimum_size(700,700)
        
        background_color = [255,255,255,255]
        background_color = [1 /255 for i in background_color]
        #glClearColor(*background_color)
        
        
    def on_key_press(self, symbol, modifiers):
        if symbol == key.A:
            print('The "A" key was pressed.')
       #     self.window.set_visible()
        elif symbol == key.LEFT:
            print('The left arrow key was pressed.')
        elif symbol == key.ENTER:
            print('The enter key was pressed.')

        pass
    
    
    def on_key_release(self, symbol, modifiers):
        pass
    
    
    def on_mouse_enter(self,x, y):
        pass

    def on_mouse_leave(self,x, y):
        pass
    
    
    def on_mouse_motion(self,x, y, dx, dy):
        pass
    
    def on_mouse_press(self,x,y,button, modifiers):
        if button == mouse.LEFT:
           print('The left mouse button was pressed.')
        elif button == mouse.RIGHT:
           print('The right mouse button was pressed.')

        pass
   
 
    def on_mouse_release(self,x, y, button, modifiers):
        pass


    def on_mouse_drag(self,x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT:
            print("mouse drag x,y,dx,dy",x,y,dx,dy)
        pass

        
    
    
    
    def on_mouse_scroll(self,x, y, scroll_x, scroll_y):
        pass
   
 #   @window.event
    def on_draw(self):
        pass
        window.clear()
        label.draw()   
   
    
    def update(self,dt):
        pass
    
# @window.event
# def on_draw():
#     window.clear()
#     label.draw()   
    
frame_rate=50

if __name__=="__main__":
    window=MyWindow(700,1000,"Connect 4",0,visible=False)
          #  win = pyglet.window.Window()
   # window.set_exclusive_mouse(True)

  #  window = pyglet.window.Window(visible=False)
# ... perform some additional initialisation
    #window.set_visible()
    
    label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')  
    
 #   
 #   event_logger = pyglet.window.event.WindowEventLogger()
 #   window.push_handlers(event_logger)
    
    
    window.set_visible()
    pyglet.app.run()
    window
    window.close()