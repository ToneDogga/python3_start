#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:42:38 2020

@author: tonedogga
"""

from collections import namedtuple
import pandas as pd
import numpy as np



#  tf model hype           # new_df[(row,name,'prediction')]=777r parameters for order prediction based on scan sales data
batch_length=4
no_of_batches=1000
no_of_repeats=4
epochs=8


# product appears on low stock report if units stock is below this number
low_stock_limit=800000   # so high it is unused currently



dash_verbose=False   #True   #False


# filenames in same directory as this programme
stock_level_query='stock_level_query.xlsx'
production_made_query='Production Schedule.xlsx'
production_made_sheet="Schedule"
production_planned_query='B Stock & Schedule Forecast.xlsx'
production_planned_sheet="Schedule"
report_savename="sales_trans_report_dict.pkl"
predictions_only_savename="_predictions.pkl"
#colesscan="Coles_scan_data_300620.xlsx"
#woolscan="Ww_scan_data_300620.xlsx"
#candatalist=["Coles_scan_data_300620.xlsx","Ww_scan_data_300620.xlsx"]
  #  report_savename="sales_trans_report_dict.pkl"
scan_df_save="coles_and_ww_invoiced_and_scanned_sales.pkl"


price_discrepencies_summary="30_day_underpriced_summary.xlsx"

 
e_scandatalist=["coles_scan_data_enhanced_sept2020.xlsx","ww_scan_data_enhanced_sept2020.xlsx"] 
transposed_datalist=["coles_scan_dataT.xlsx","ww_scan_dataT.xlsx"]  

extra_scan_data=["chutneys_UPSPW.xlsx","sauces_UPSPW.xlsx","jams_UPSPW.xlsx","chutney_units.xlsx","sauces_units.xlsx","jams_units.xlsx"]

e_scandata_plotqueries=[[('10','retailer'),('0','variety')],
                        [('12','retailer'),('0','variety')],
                        [('10','retailer'),('1','variety')],
                        [('12','retailer'),('1','variety')],
                        [('1','variety')],
                        [('2','variety')],
                        [('3','variety')],

                        [('10','retailer'),('2','variety')],
                        [('12','retailer'),('2','variety')],
                        [('10','retailer'),('3','variety')],
                        [('12','retailer'),('3','variety')],
                        [('10','retailer'),('4','variety')],
                        [('12','retailer'),('4','variety')],
                        
                        [('10','retailer'),('9','variety')],
                        [('12','retailer'),('9','variety')],
                        [('10','retailer'),('10','variety')],
                        [('12','retailer'),('10','variety')],
                        [('10','retailer'),('11','variety')],
                        [('12','retailer'),('11','variety')],
                        [('10','retailer'),('14','variety')],
                        [('12','retailer'),('14','variety')],
                        [('10','retailer'),('15','variety')],
                        [('12','retailer'),('15','variety')],
                        [('10','retailer'),('16','variety')],
                        [('12','retailer'),('16','variety')],
 
                        
                   #     [('10','retailer'),('3','variety')],
                   #     [('12','retailer'),('4','variety')],
                        
                        [('10','retailer'),('30','variety')],
                        [('12','retailer'),('30','variety')],
                        [('10','retailer'),('31','variety')],
                        [('12','retailer'),('31','variety')],
                        
                        [('10','retailer'),('50','variety')],
                        [('12','retailer'),('50','variety')],
                        [('10','retailer'),('51','variety')],
                        [('12','retailer'),('51','variety')],
                        [('10','retailer'),('52','variety')],
                        [('12','retailer'),('52','variety')],
                        [('10','retailer'),('53','variety')],
                        [('12','retailer'),('53','variety')],

                        [('10','retailer'),('90','variety')],
                        [('12','retailer'),('90','variety')],
                        [('10','retailer'),('91','variety')],
                        [('12','retailer'),('91','variety')],
                        [('10','retailer'),('92','variety')],
                        [('12','retailer'),('93','variety')],
                        [('10','retailer'),('93','variety')],
                        [('12','retailer'),('93','variety')],
 
         
                        [('10','productgroup')],
                        [('11','productgroup')],
                        [('12','productgroup')],
                        [('13','productgroup')],

                    #    [('12','retailer'),('10','productgroup')],

              
                        [('10','retailer'),('10','productgroup')],
                        [('12','retailer'),('10','productgroup')],

                        [('10','retailer'),('11','productgroup')],
                        [('12','retailer'),('11','productgroup')],

                        [('10','retailer'),('12','productgroup')],
                        [('12','retailer'),('12','productgroup')],

                        [('10','retailer'),('13','productgroup')],
                        [('12','retailer'),('13','productgroup')],

                        [('10','retailer'),('14','productgroup')],
                        [('12','retailer'),('14','productgroup')],

                        [('10','retailer'),('15','productgroup')],
                        [('12','retailer'),('15','productgroup')],

                        [('10','retailer'),('16','productgroup')],
                        [('12','retailer'),('16','productgroup')],

                        [('10','retailer'),('17','productgroup')],
                        [('12','retailer'),('17','productgroup')],


  

                        [('1','brand')],  #'    augmented_sales_df.to_pickle(dd.sales_df_augmented_savename,protocol=-1)')],
                        [('2','brand')],
                        [('3','brand')],
                        [('4','brand')],
                        [('7','brand')],
                        [('8','brand')],
                        [('13','brand')],
                        [('18','brand')]]
                       
                     #   [('10','retailer'),('12','productgroup')],
                  #      [('12','retailer'),('14','productgroup')]]


customers_to_plot_together=['FLNOR',"FLFRE","FLSTI", "FLPAS","FLBRI"]   #"FLDAW","GLENORC","IGAATH"]  #"FLPAS","FLMIT","FLBRI"]
scaling_point_week_no=51

e_scandata_number_of_weeks=53  # number of weeks back from latest week

brand_index_weeks_going_back=104   # for brand index

weeks_offset=3   # weeks to shift invoiced sales to align with scanned sALES

weeks_rolling_mean=3  # week mrolling ean on invoiced sales

sales_df_savename="sales_trans_df.pkl"
sales_df_augmented_savename="sales_trans_df_augmented.pkl"
sales_df_complete_augmented_savename="sales_trans_df_complete_augmented.pkl"
 #  sales_df_augmented_savename="sales_trans_df_augmented.pkl"

price_df_savename="price_df.pkl"


distribution_list_oneyear='distribution_list_oneyear.pkl'
distribution_list_lastoneyear='distribution_list_lastoneyear.pkl'
distribution_list_twoyear='distribution_list_twoyear.pkl'
distribution_list_thisyear_minus_lastyear='distribution_list_thisyear_minus_lastyear.pkl'
distribution_list_thisyear_minus_lastyear_percent="distribution_list_thisyear_minus_lastyear_percent.pkl"
filenames=["allsalestrans2020.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]   #"allsalestrans190520.xlsx", "allsalestrans101120.xlsx"
   

product_groups_only=["10"]  #,"11","12","13","14","15","16","17"]   #,"18"]
spc_only=['048']   #,'028','048']   #,'080','020',"028",'030',"038",'040',"048",'050','060','070',"122","107"]   #,"028"]   #,"038","048","028","080","020","030","040']

max_slope=0.16
min_slope=-0.1

min_size_for_trend_plot=6

# tuple start_date, end_date
#pareto_dates=[(pd.to_datetime("01/03/19",format="%d/%m/%y"),pd.to_datetime("01/09/19",format="%d/%m/%y")),(pd.to_datetime("01/03/20",format="%d/%m/%y"),pd.to_datetime("01/09/20",format="%d/%m/%y")),(pd.to_datetime("01/07/19",format="%d/%m/%y"),pd.to_datetime("30/06/20",format="%d/%m/%y")),(pd.to_datetime("01/07/20",format="%d/%m/%y"),pd.to_datetime("30/06/21",format="%d/%m/%y"))]
pareto_dates=[(pd.to_datetime("01/07/19",format="%d/%m/%y"),pd.to_datetime("30/06/20",format="%d/%m/%y")),(pd.to_datetime("01/07/20",format="%d/%m/%y"),pd.to_datetime("30/06/21",format="%d/%m/%y"))]




# moving average total days used in prediction
#mats=7

# moving average for weeks to calculate a rolling average on scanned data
mat=4      # weeks mat for product sales smoothing
mat2=52   # weeks mat for customer reports

#can_data_files=["jam_scan_data_2020.xlsx","cond_scan_data_2020.xlsx","sauce_scan_data_2020.xlsx"]
#can_dict_savename="scan_dict.pkl"


price_width=60  # columns to import from salestrans.xlsx on price sheet


productgroup_dict={"01":"30g glass",
                   "02":"40g glass",
                   "041":"15g Sterling",
                   "042":"30g Sterling",
                   "043":"50g Sterling",
                   "05":"14g SIS",
                   "10":"Jams 250ml glass jar",
                   "11":"Sauce 300ml glass bottle",
                   "12":"Dressings 300ml glass bottle",
                   "13":"Condiments 250ml glass jar",
                   "14":"Meal bases 250ml glass jar",
                   "15":"Condiments for cheese 150ml glass jar",
                   "16":"Traditional condiments 150ml glass jar",
                   "17":"Mustards 150ml glass jar",
                   "18":"Olive Oil 500ml",
                   "31":"2 Litre",
                   "999":"999"}

productgroups_dict={1:"30g glass",
                   2:"40g glass",
                   41:"15g Sterling",
                   42:"30g Sterling",
                   43:"50g Sterling",
                   5:"14g SIS",
                   10:"Jams 250ml glass jar",
                   11:"Sauce 300ml glass bottle",
                   12:"Dressings 300ml glass bottle",
                   13:"Condiments 250ml glass jar",
                   14:"Meal bases 250ml glass jar",
                   15:"Condiments for cheese 150ml glass jar",
                   16:"Traditional condiments 150ml glass jar",
                   17:"Mustards 150ml glass jar",
                   18:"Olive Oil 500ml",
                   31:"2 Litre",
                   999:""}


series_type_dict={0:0, #"0 baseline",
             1:1,    #"1 incremental",
             2:2,   #"2 total",
             3:3,    #"3 invoiced",
             4:4,    #"4 invoiced_shifted",
             6:6,   #"6 scanned 4wk mt",
             7:7,   #"7 invoiced shifted 4wk mt",
             8:8,    # invoiced_prediction
             9:9}    #"9 promo flag"}




salesrep_dict={"14":"Brendan Selby",
               "8":"Noel Lotorto",
               "33":"Howard Lee",
               "34":"Russell Heyzer",
               "35":"Stephen Wood",
               "36":"Annette Paech",
               "37":"Miles Rafferty",
               "39":"Sophia Simos",
               "11":"Coles and WW",
               "999":""}
               


spc_dict={122:"Harris farms",
          107:"Lite n easy",
          10:"woolworths",
          12:"coles",
          88:"SA stores",
          80:"SA distributors",
          20:"NSW distributors",
          40:"QLD distributors",
          30:"VIC distributors",
          38:"VIC stores",
          28:"NSW stores",
          48:"QLD stores",
          50:"WA distributors",
          70:"TAS distributors",
          92:"Shop",
          95:"Online shop",
          999:""}


report = namedtuple("report", ["name", "report_type","cust","prod"])

report_type_dict={0:"dictionary",
                       3:"dataframe",
                       5:"spreadsheet",
                       6:"pivottable",
                       8:"chart_filename"}



   #report = namedtuple("report", ["name", "report_type","cust","prod"])
 
#scandata = namedtuple("scandata", ["market", "product","measure"])

market_rename_dict={"AU Woolworths scan":10,
                    "AU Coles Group scan":12}

market_rename_dict2={"AU Woolworths scan":"woolworths",
                    "AU Coles Group scan":"coles"}




