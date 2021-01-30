#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:23:27 2020

@author: tonedogga
"""

# import sales
# import scan
# import price
# import scheduler
# import production
import ave_dash2
import animate_plots
# import stock_cost
# import predict_majors2
# import interactive_query
import odbc_firehose
import bbtv_gui

import pandas as pd

class bbtv1_class():
   def __init__(self):   #,in_value):
      # self.dash2_init="dash2_init called"
      # print(self.dash2_init)
      # print("in value",in_value)
       self.firehose=odbc_firehose.odbc_firehose_class()
       
       self.frontend=bbtv_gui.gui
    #   self.window_gui=bbtv_frontend
      
      # self.sales=sales.sales_class(pd.DataFrame([]))
      # self.scan=scan.scan_class()
      # self.price=price.price_class()
      # self.production=production.production_class()
       self.ave_predict=ave_dash2.ave_class()
      # self.scheduler=scheduler.scheduler()
       self.animate=animate_plots.animate_engine()
      # self.stock=stock_cost.stock_cost()
     #  self.predict=predict_majors2.predict_majors()
 
