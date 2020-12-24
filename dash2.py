#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:33:31 2020

@author: tonedogga
"""

import pandas as pd
import numpy as np


import os
import sys

import calendar
import xlsxwriter

import xlrd
import datetime as dt
from datetime import datetime,timedelta
from pandas.plotting import scatter_matrix

from matplotlib import pyplot, dates
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

import multiprocessing
import time
import joblib
import warnings
import pickle   

import sklearn.linear_model
import sklearn.neighbors

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from pandas.plotting import scatter_matrix


from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)>0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.autograph.set_verbosity(0, False)
import subprocess as sp

from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"

# =============================================================================
# if dd.dash_verbose==False:
#      tf.autograph.set_verbosity(0,alsologtostdout=False)   
#    #  tf.get_logger().setLevel('INFO')
# else:
#      tf.autograph.set_verbosity(1,alsologtostdout=True)   
# 
# =============================================================================



tf.config.run_functions_eagerly(False)


 
####################################################################################

import dash2_root
import dash2_dict as dd2


dash=dash2_root.dash2_class()   #"in_dash value")   # instantiate a salestrans_df

#############################################################################################
os.chdir("/home/tonedogga/Documents/python_dev")
cwdpath = os.getcwd()
#print(cwdpath)



warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format

#  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

visible_devices = tf.config.get_visible_devices('GPU') 





class SABusinessCalendar(AbstractHolidayCalendar):
   rules = [
     Holiday('New Year', month=1, day=1), #observance=sunday_to_monday),
     Holiday('Australia Day', month=1, day=26, observance=sunday_to_monday),
   
     Holiday('Anzac Day', month=4, day=25),
     Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
     Holiday('Easter Monday', month=1, day=1, offset=[Easter(),Day(+1)]),    

   
     Holiday('October long weekend', month=10, day=1, observance=sunday_to_monday),
     Holiday('Christmas', month=12, day=25, observance=nearest_workday),
     Holiday('Boxing day', month=12, day=26, observance=nearest_workday),
     Holiday('Dec27 shutdown', month=12, day=27),   #, observance=nearest_workday)    

     Holiday('Proclamation day', month=12, day=28, observance=nearest_workday),
     Holiday('Dec29 shutdown', month=12, day=29),   #, observance=nearest_workday)    
     Holiday('Dec30 shutdown', month=12, day=30),   #, observance=nearest_workday)     
     Holiday('Dec31 shutdown', month=12, day=31)   #, observance=nearest_workday)     

   ]
      




def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./dash2_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

#output_dir = log_dir("dashboard")





def main():
    tmp=sp.call('clear',shell=True)  # clear screen 'use 'clear for unix, cls for windows
    plot_output_dir = log_dir("dash2")
    dd2.dash2_dict['sales']['plot_output_dir']=plot_output_dir
    
    print("\nDash2-Beerenberg analyse, visualise and predict- By Anthony Paech 17/11/20")
    print("=================================================================================================\n")       
    
    print("Python version:",sys.version)
    print("Current working directory",cwdpath)
    print("plot_output dir",plot_output_dir)
    print("data save directory",dd2.dash2_dict['sales']['save_dir'])
    print("\ntensorflow:",tf.__version__)
    #print("TF2 eager exec:",tf.executing_eagerly())      
    print("keras:",keras.__version__)
    print("numpy:",np.__version__)
    print("pandas:",pd.__version__)
    print("matplotlib:",mpl.__version__)      
    print("sklearn:",sklearn.__version__)         
    print("\nnumber of cpus : ", multiprocessing.cpu_count())            
    print("tf.config.get_visible_devices('GPU'):\n",visible_devices)
 #   print("\n",pd.versions(),"\n")
    print("\n=================================================================================================\n")       
     
    
   
    refresh=(input("Refresh transaction data from excel? (y/n)").lower()=='y')  
    shortcut=(input("Take the shortcut? (y/n)").lower()=='y')  
   
 
    todays_date = dt.date.today()   #time.now()  #.normalize()   #dt.date.today()
    other_todays_date=dt.datetime.now()
    SA_BD = CustomBusinessDay(calendar=SABusinessCalendar())
    # start_schedule_day_offset=0
      
    date_choices=pd.bdate_range(todays_date,todays_date+timedelta(dd2.dash2_dict['scheduler']['maximum_days_into_schedule_future']),normalize=True,freq=SA_BD)
    date_choices_dict={}
     
    for d in date_choices:
        i=int((d-other_todays_date).days+1)
        date_choices_dict[i]=d
 
    
    #if False: #we are testing
   # start_schedule_day_offset=dd2.dash2_dict['scheduler']['default_start_schedule_days_delay']
   # start_schedule_date=date_choices_dict[int(start_schedule_day_offset)]
    if True:    
        #print("dcd=",date_choices_dict)    
        incorrect=True
        while incorrect:    
            start_schedule_day_offset=input("Days ahead from today to start scheduling manufacturing?")
            if isinstance(start_schedule_day_offset,str):
                try:
                    start_schedule_day_offset=int(start_schedule_day_offset)
                except:
                    pass
                else:
                    if isinstance(start_schedule_day_offset,int):
                        try:
                            start_schedule_date=date_choices_dict[int(start_schedule_day_offset)]
                        except KeyError:
                            print("start date is weekend or invalid.")
                            pass
                        else:
                            incorrect=False

    
    
    if refresh:
        print("\nLoad sales data from",len(dd2.dash2_dict['sales']['in_files']),"excel files in current working directory (",cwdpath,")....")
        sales_df=dash.sales.load_from_excel(dd2.dash2_dict['sales']['in_dir'],dd2.dash2_dict['sales']['in_files'])
        sales_df=dash.sales.preprocess_sc(sales_df,dd2.dash2_dict['sales']['rename_columns_dict'])
        first_date,latest_date=dash.sales.report(sales_df)
        if dash.sales.save(sales_df,dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['savefile']):
            print("sales_df",sales_df.shape,"loaded, masked, pickled and saved to",dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['savefile'],"\n")
            pass
        else:
            print("sales_df save failure.")
            
    
        price_df=dash.price.load_from_excel(dd2.dash2_dict['price']['in_dir'],dd2.dash2_dict['price']['in_file'])  
        price_df=dash.price.preprocess(price_df)
        if dash.price.save(price_df,dd2.dash2_dict['price']['save_dir'],dd2.dash2_dict['price']['savefile']):
            print("price_df",price_df.shape,"loaded, pickled and saved to",dd2.dash2_dict['price']['save_dir']+dd2.dash2_dict['price']['savefile'],"\n")
            pass
        else:
            print("price_df save failure.")
   
        scan_df=dash.scan.load_scan_data_from_excel(dd2.dash2_dict['scan']['in_dir'],dd2.dash2_dict['scan']['scan_data_list'],dd2.dash2_dict['scan']['transposed_scan_data_list'])  
        if dash.scan.save(scan_df,dd2.dash2_dict['scan']['save_dir'],dd2.dash2_dict['scan']['savefile']):
            print("scan_df",scan_df,scan_df.shape,"loaded, masked and saved to",dd2.dash2_dict['scan']['save_dir']+dd2.dash2_dict['scan']['savefile'],"\n")
        #    print("\n",scan_df.T)
            pass
        else:
            print("scan_df save failure.")
   
        scan_monthly_df=dash.scan.load_scan_monthly_data_from_excel(dd2.dash2_dict['scan']['in_dir'],dd2.dash2_dict['scan']['scan_monthly_data_list'],weeks_back=53)  
        if dash.scan.save(scan_monthly_df,dd2.dash2_dict['scan']['save_dir'],dd2.dash2_dict['scan']['monthlysavefile']):
            print("scan_monthly_df",scan_monthly_df.shape,"loaded, masked and saved to",dd2.dash2_dict['scan']['save_dir']+dd2.dash2_dict['scan']['monthlysavefile'],"\n")
            pass
        else:
            print("scan_monthly_df save failure.")
         
   
    
   
    else:
        
        sales_df=dash.sales.load(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['savefile'])
        sales_df=dash.sales._preprocess(sales_df,dd2.dash2_dict['sales']['rename_columns_dict'])
        first_date,latest_date=dash.sales.report(sales_df)
        scan_df=dash.scan.load(dd2.dash2_dict['scan']['save_dir'],dd2.dash2_dict['scan']['savefile'])  
        scan_monthly_df=dash.scan.load(dd2.dash2_dict['scan']['save_dir'],dd2.dash2_dict['scan']['monthlysavefile'])  




    start_timer = time.time()
   # print("new sales_df=\n",new_sales_df)  
    dd2.dash2_dict['sales']['sales_df']=sales_df.copy()
    price_df=dash.price.load(dd2.dash2_dict['price']['save_dir'],dd2.dash2_dict['price']['savefile'])
    aug_sales_df,promo_pivot_df=dash.price.flag_promotions(sales_df,price_df,plot_output_dir)
    scan_monthly_df=dash.scan.preprocess_monthly(scan_monthly_df)

    
    #if not refresh:
    first_date,latest_date=dash.sales.report(aug_sales_df)  
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=    
  #  print("aug sales df=\n",aug_sales_df)
    print("Run queries on loaded sales_df data:",aug_sales_df.shape,"(",first_date.strftime('%d/%m/%Y'),"to",latest_date.strftime('%d/%m/%Y'),")") 
 #   query_dict=dash.sales.query.queries(sales_df)   #"sales query infile4","g")
    query_dict=dash.sales.query.queries(aug_sales_df)   #"sales query infile4","g")

    with open(dd2.dash2_dict['sales']["save_dir"]+dd2.dash2_dict['sales']["query_dict_savefile"], 'wb') as f:
        pickle.dump(query_dict,f,protocol=-1)
    print("query dict keys=\n",query_dict.keys())
    for k,v in query_dict.items():
        print(k,"\n",v,v.shape)
        
  #  print("scan_monthly_df=\n",scan_monthly_df) #.shape)   #,"\n",scan_monthly_df.T)   #.shape)   
    print("Run queries on loaded scan_monthly_df data:",scan_monthly_df.shape) 
    scan_monthly_dict=dash.scan.scan_monthly_queries(scan_monthly_df)    
    print("scan monthly dict keys=\n",scan_monthly_dict.keys())
    for k,v in scan_monthly_dict.items():
        print(k,"\n",v.shape)
        
        
        
   # print(query_dict)
    print("augmented sales_df=\n",aug_sales_df.shape)
  #  print("price_df=\n",price_df)
    print("scan_df=\n",scan_df.shape)
    print("scan_monthly_df=\n",scan_monthly_df.shape)   #,"\n",scan_monthly_df.T)   #.shape)
  #  print("scan monthly - available_retailers,available_products",available_retailers,available_products)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #original_stdout = sys.stdout # Save a reference to the original standard output

    stock_report_df=dash.production.load_from_excel(dd2.dash2_dict['production']['in_dir'])
    dash.production.report(stock_report_df,dd2.dash2_dict['production']['in_dir'])
    dash.stock.stock_summary(plot_output_dir)
    
    dash.scheduler.display_schedule(plot_output_dir)

    all_raw_dict=dash.sales.summary(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['raw_savefile'])
  #  dash.sales.pivot.report(all_raw_dict['raw_all'],plot_output_dir)
    dash.price.report(aug_sales_df,promo_pivot_df,plot_output_dir)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("calculate and plot sales trends for every combination of customer and product....")
    original_stdout = sys.stdout 
    with open(plot_output_dir+dd2.dash2_dict['sales']['print_report'], 'w') as f:
       sys.stdout = f 
       print("\nDash2 started:",dt.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S %d/%m/%Y'),"\n")
       
       dash.production.report(stock_report_df,dd2.dash2_dict['production']['in_dir'])
       dash.stock.stock_summary(plot_output_dir)
   #     print(stock_df.to_string(),"\n")
       dash.scheduler.display_schedule(plot_output_dir)
       
       all_raw_dict=dash.sales.summary(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['raw_savefile'])
       dash.sales.pivot.report(all_raw_dict['raw_all'],plot_output_dir)
       dash.price.report(aug_sales_df,promo_pivot_df,plot_output_dir)
       
    sys.stdout=original_stdout
 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   # sales trans data    
   
    if not shortcut:   #False: #True:  
        dash.sales.pivot.distribution_report_dollars('last_today_to_365_days',query_dict['last_today_to_365_days'],plot_output_dir,trend=True)
      
        
        dash.sales.plot.mat(all_raw_dict,dd2.dash2_dict['sales']['annual_mat'],latest_date,plot_output_dir)
        dash.sales.plot.mat(query_dict,dd2.dash2_dict['sales']['annual_mat'],latest_date,plot_output_dir)
        #dash.sales.plot.mat_stacked_product(query_dict,dd2.dash2_dict['sales']['annual_mat'],latest_date,plot_output_dir)
       
       
        dash.sales.plot.yoy_dollars(query_dict,7,latest_date,plot_output_dir)
        dash.sales.plot.yoy_units(query_dict,7,latest_date,plot_output_dir)
        # dash.sales.plot.compare_customers(query_dict,dd2.dash2_dict['sales']['annual_mat'],dd2.dash2_dict['sales']['plots']['customers_to_plot_together'],latest_date,plot_output_dir)
        dash.sales.plot.p_compare_customers(query_dict,dd2.dash2_dict['sales']['annual_mat'],dd2.dash2_dict['sales']['plots']['customers_to_plot_together'],latest_date,plot_output_dir)
        dash.sales.plot.p_compare_products(query_dict,dd2.dash2_dict['sales']['annual_mat'],dd2.dash2_dict['sales']['plots']['products_to_plot_together'],latest_date,plot_output_dir)
    
    
    
      
        dash.sales.plot.pareto_customer(query_dict,latest_date,plot_output_dir)
        dash.sales.plot.pareto_product_dollars(query_dict,latest_date,plot_output_dir)
        dash.sales.plot.pareto_product_units(query_dict,latest_date,plot_output_dir)
     
        print("distribution report dollars + trend sort")
        original_stdout = sys.stdout 
        with open(plot_output_dir+dd2.dash2_dict['sales']['print_report'], 'a') as f:
           sys.stdout = f 
           dash.sales.pivot.distribution_report_dollars('last_today_to_365_days',query_dict['last_today_to_365_days'],plot_output_dir,trend=True)
       
        sys.stdout=original_stdout 
        
        dash.sales.pivot.distribution_report_dollars('last_365_to_731_days',query_dict['last_365_to_731_days'],plot_output_dir,trend=False)
      
        this_year=query_dict['last_today_to_365_days'].copy()
        last_year=dash.sales.pivot.negative(query_dict['last_365_to_731_days'])   # multiple all athe salesval by -1 
        change_year=pd.concat((this_year,last_year),axis=0)
        dash.sales.pivot.distribution_report_dollars('change dollars',change_year,plot_output_dir,trend=False)
    
        dash.sales.pivot.distribution_report_units('last_today_to_365_days',query_dict['last_today_to_365_days'],plot_output_dir)
        dash.sales.pivot.distribution_report_units('last_365_to_731_days',query_dict['last_365_to_731_days'],plot_output_dir)
       
        this_year=query_dict['last_today_to_365_days'].copy()
        last_year=dash.sales.pivot.negative(query_dict['last_365_to_731_days'])   # multiple all athe salesval by -1 
        change_year=pd.concat((this_year,last_year),axis=0)
        dash.sales.pivot.distribution_report_units('change units',change_year,plot_output_dir)
    
        dash.sales.pivot.distribution_report_dates('last_today_to_365_days',query_dict['last_today_to_365_days'],plot_output_dir)
 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
  #  Scan data 
    
        dash.scan.brand_index(scan_df,plot_output_dir)  
         
        dash.scan.plot_scan_weekly(scan_df,plot_output_dir)  
        dash.scan.plot_scan_monthly_dict(scan_monthly_dict,plot_output_dir)
   
 
    dash.sales.predict.predict_order(scan_df,aug_sales_df,latest_date,plot_output_dir)
    dash.predict.predict_majors_orders(plot_output_dir) 
    
    final_schedule,saved_start_schedule_date=dash.scheduler.visualise_schedule(start_schedule_day_offset,start_schedule_date,plot_output_dir)

    original_stdout=sys.stdout 
    with open(plot_output_dir+dd2.dash2_dict['sales']['print_report'], 'a') as f:
       sys.stdout = f 
       dash.sales.predict._repredict_rfr(plot_output_dir)
       
       dash.scheduler.display_and_export_final_schedule(final_schedule,saved_start_schedule_date,plot_output_dir)
    sys.stdout=original_stdout
       
    dash.scheduler.animate_plots(gif_duration=4,mp4_fps=10,plot_output_dir=plot_output_dir)
 
    query_dict2=dash.sales.query.queries(dd2.dash2_dict['sales']['sales_df'])    
  
    
    dash.animate.animate_brand_index(plot_output_dir+dd2.dash2_dict['sales']['plots']["animation_plot_dump_dir"],plot_output_dir,mp4_fps=11)
    dash.animate.animate_query_dict(query_dict2,plot_output_dir+dd2.dash2_dict['sales']['plots']["animation_plot_dump_dir"],plot_output_dir,mp4_fps=11)
 
   
  
 #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
   # actual vs expected prediction
         
  
    dash.ave_predict.actual_vs_expected(query_dict,plot_output_dir)
   


 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
   
    end_timer = time.time()
    print("\nDash2 finished:",dt.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S %d/%m/%Y'))
    print("Dash2 total runtime:",round(end_timer - start_timer,2),"seconds.\n")





main()

