#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:55:55 2020

@author: tonedogga
"""
import os
import pandas as pd
import pickle
import dash2_dict as dd2
os.chdir("/home/tonedogga/Documents/python_dev")
if True:
    with open(dd2.dash2_dict['sales']["save_dir"]+dd2.dash2_dict['sales']["query_dict_savefile"], 'rb') as f:
        query_dict=pickle.load(f)
    print("query dict keys=\n",query_dict.keys())
#    for k,v in query_dict.items():
#        print(k,"\n",v,v.shape)
print(query_dict["Woolworths (010) all"].head(1600).sum())