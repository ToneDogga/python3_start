#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:59:58 2020

@author: tonedogga
"""
import os
import pandas as pd
import dash2_dict as dd2

os.chdir("/home/tonedogga/Documents/python_dev")
cwdpath = os.getcwd()
stocks=pd.read_pickle(dd2.dash2_dict['scheduler']['schedule_savedir']+dd2.dash2_dict['scheduler']['schedule_savefile'])
print(stocks) 