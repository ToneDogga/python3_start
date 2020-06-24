#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:52:03 2020

@author: tonedogga
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

report = namedtuple("report", ["name", "report_type"])


# report_type_dict=dict({0:"dictionary",
#                        3:"dataframe",
#                        5:"spreadsheet",
#                        6:"pivottable",
#                        8:"chart_filename"})





report_savename="sales_trans_report_dict.pkl"

try:
    with open(report_savename,"rb") as f:
        report_dict=pickle.load(f)
except:
    print(report_savename,"not found")

print(report_dict.keys(),"\n")
report_type_dict=report_dict[report("report_type_dict",0)]
print("report type dict",report_type_dict)

for key in report_dict.keys():
    print(key[0],key[1])
    if key[1]==8:
        print("\nChart")
        pickle.load(open(report_dict[key],'rb')).figure
    elif key[1]==6:
        print("\nPivot table-",key)
        print(report_dict[key])
    elif key[1]==5:
        print("\nSpreadsheet-",key)
    elif key[1]==3:
        print("\nDataFrame-",key)
        print(report_dict[key])
        
 
        
       # fig2.figure
    #else:    
    #    print(key,report_dict[key])
