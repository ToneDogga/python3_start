#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:55:24 2020

@author: tonedogga
"""
 
import glob
import os
import shutil

os.chdir("/home/tonedogga/Documents/python_dev") 
plot_dump_dir="./dash2_outputs/dash2-run-20201212065807/animation_plot_dump_dir/"
plot_output_dir="./dash2_outputs/dash2-run-20201212065807/"

files = glob.glob(plot_dump_dir+"*.mp4")  #dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+'*.png')
#  print("1files to del:",files)
  # copy mp4 files over to main output dir
#print(files)  
for item in files:
    #  print(item)
      filename = os.path.basename(item)
   #   print(filename,item)
      shutil.copy(item, os.path.join(plot_output_dir, filename))
   
