#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:44:26 2020

@author: tonedogga
"""
""
import pandas as pd
from pandas.tseries.frequencies import to_offset
import dash2_dict as dd2


class production_class(object):
   def __init__(self):
       #self.scan_init="scan_init called"
      # print(self.scan_init)
       pass
 
    
 
   def load_from_excel(self,in_dir):
        print("load production to",in_dir)
       
        print("load:",in_dir+dd2.dash2_dict['production']['stock_level_query'])
        stock_df=pd.read_excel(in_dir+dd2.dash2_dict['production']['stock_level_query'])    # -1 means all rows   
       # with open("stock_level_query.pkl","wb") as f:
       #     pickle.dump(stock_df, f,protocol=-1)
        #except:
        #    pass        
        #print("stock df size=",stock_df.shape,stock_df.columns)
        #
            
        stock_df['end_date']=stock_df['lastsalesdate']+ pd.Timedelta(90, unit='d')
        #print(end_date)
        #print("ysdf=",sales_df)
        stock_df['recent']=stock_df['end_date']>pd.Timestamp('today')
        
        
        
        #print("stock_df=\n",stock_df)
        #stock_df=stock_df[(stock_df['qtyinstock']<=2000) & (stock_df['recent']==True) & ((stock_df['productgroup']>=10) & (stock_df['productgroup']<=17))]  # | (stock_df['productgroup']==12) | (stock_df['productgroup']==13) | (stock_df['productgroup']==14) | (stock_df['productgroup']<=17))]
        stock_df=stock_df[(stock_df['recent']==True) & (stock_df['qtyinstock']<=dd2.dash2_dict['production']['low_stock_limit']) & ((stock_df['productgroup']>=10) & (stock_df['productgroup']<=17))]  # | (stock_df['productgroup']==12) | (stock_df['productgroup']==13) | (stock_df['productgroup']==14) | (stock_df['productgroup']<=17))]
                        
        stock_report_df=stock_df[['productgroup','code','lastsalesdate','qtyinstock']].sort_values(['productgroup','qtyinstock'],ascending=[True,True])
        
        stock_report_df.replace({'productgroup':dd2.productgroups_dict},inplace=True)
        self._save_stock(stock_report_df)
        return stock_report_df
    
    
        
   def report(self,stock_report_df,in_dir):    
      #  print("Low stock report (below",dd.low_stock_limit,"units)\n",stock_report_df.to_string())
        print("\n============================================================================\n")  
        print("Low stock report:\n",stock_report_df.to_string())
        
        print("load:",in_dir+dd2.dash2_dict['production']['production_made_query'])
        production_made_df=pd.read_excel(in_dir+dd2.dash2_dict['production']['production_made_query'],sheet_name=dd2.dash2_dict['production']['production_made_sheet'])    # -1 means all rows   
        production_made_df=production_made_df[['to_date','jobid','code','qtybatches','qtyunits']].sort_values('to_date',ascending=True)
        print("\nProduction recently made:\n",production_made_df.tail(50))
        self._save_PM(production_made_df)
        
        print("load:",in_dir+dd2.dash2_dict['production']['production_planned_query'])
        production_planned_df=pd.read_excel(in_dir+dd2.dash2_dict['production']['production_planned_query'],sheet_name=dd2.dash2_dict['production']['production_planned_sheet'])    # -1 means all rows   
        
        production_planned_df['future']=production_planned_df['to_date']>=pd.Timestamp('today')
        production_planned_df=production_planned_df[(production_planned_df['future']==True)]
                        
        production_planned_df=production_planned_df[['to_date','jobid','code','qtybatches','qtyunits']].sort_values('to_date',ascending=True)
        print("\nProduction planned:\n",production_planned_df.head(50))
        self._save_PP(production_planned_df)
        print("\n============================================================================\n")  
              
        return
    
    
    
 
        
   def preprocess(self):
       # rename columns
       print("preprocess production df")
       return

       
   def _save_stock(self,df):
       print("save stock")
       df.to_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['SOH_savefile'],protocol=-1)
       return
 
       
   def _save_PP(self,df):
       print("save PP")
       df.to_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PP_savefile'],protocol=-1)
       return
     
       
   def _save_PM(self,df):
       print("save PM")
       df.to_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PM_savefile'],protocol=-1)
       return
 
    