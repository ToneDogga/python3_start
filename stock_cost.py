#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:26:13 2020

@author: tonedogga
"""

import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path

import dash2_dict as dd2
  


class stock_cost(object):    
   def _load_SOH(self):
       filenames=sorted(glob.glob(dd2.dash2_dict['production']['save_dir']+'SOH_[*.pkl')) 
       stock_level_dict={}
       for f in filenames:
           g=f.split('[')[1].split("]")[0]
           stock_level_dict[g]=pd.read_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['SOH_savefile'])
       return stock_level_dict   
       
 
       
#   def _load_PP(self):
#       return pd.read_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PP_savefile'])
       
     
       
   def _load_PM(self):
       return pd.read_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PM_savefile'])
       


   def _load_sales(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            return pd.read_pickle(save_dir+savefile)
        else:
            print("load sales_df error.")
            return
      
        
      
   def load(self):
        stock_levels_df=self._load_SOH()
     #   print("stock levels=\n",stock_levels_df)
        product_made_df=self._load_PM()
       # print("product made=\n",product_made_df)
       
        sales_df=self._load_sales(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['raw_savefile'])
        sales_df.sort_index(axis=0,ascending=True,inplace=True)
        #sales_df=dash.sales._preprocess(sales_df,dd2.dash2_dict['sales']['rename_columns_dict'])
      #  print("sales=\n",sales_df)
        return stock_levels_df,product_made_df,sales_df



   def bring_together(self,stock_levels_dict,product_made_df,sales_df):
        #print("stock levels=\n",stock_levels_dict)
     #   print("product made=\n",product_made_df)
        #print("sales=\n",sales_df)
        stocktake_key=max(stock_levels_dict.keys())
        stocktake_date=pd.to_datetime(stocktake_key)+pd.offsets.Day(0)

      #  print("most recent stocktake date:",stocktake_date)
 
        stock_levels=stock_levels_dict[stocktake_key][['code','qtyinstock','lastsalesdate']].copy()
        stock_levels.rename(columns={"code":"product","lastsalesdate":"stockqtydate"},inplace=True)
        
        stock_levels.set_index('product',inplace=True)
     #   new_stock_levels=stock_levels.rename({"code":"product"}).copy()
     #   stock_levels=new_stock_levels[['code','qtyinstock']]
       # stock_levels=stock_levels[['code','qtyinstock']].set_index('code')
     #   print(stock_levels)
        recent_product_made=product_made_df[product_made_df['to_date']>=stocktake_date].copy()   #.set_index('code')   #,drop=False)
        recent_product_made=recent_product_made.rename(columns={"code":"product","qtyunits":"qtymade","to_date":"made_date"})
      #  print("+product made since then:\n",recent_product_made)
        recent_product_group=recent_product_made[['product','qtymade','made_date']].groupby(['product']).agg({"qtymade":"sum","made_date":"min"})
     #   print("+product group since then:\n",recent_product_group)
        recent_sales_df=sales_df[sales_df.index>=stocktake_date].copy()
     #   print("recent sales=\n",recent_sales_df)
        recent_sales_group=recent_sales_df[['product','qty','date']].groupby(['product']).agg({"qty":"sum","date":"max"})
      #  print("product sold since then:\n",recent_sales_group)   #.to_string())
        rsg=recent_sales_group.rename(columns={"qty":"qtysold","date":"sold_date"})
      #  print("- stock sold\n",rsg)
        final=stock_levels.join([recent_product_group,rsg])
        final['qtyinstock']=final['qtyinstock'].replace(np.nan,0)
        final['qtymade']=final['qtymade'].replace(np.nan,0)
        final['qtysold']=final['qtysold'].replace(np.nan,0)
       # final['sold_date']=final['sold_date'].replace(np.nan,0)
     #   final.sort_index(ascending=True,axis=0,inplace=True)
        final['qty']=final['qtyinstock']+final['qtymade']-final['qtysold']
        final.sort_values(['qty'],axis=0,ascending=[True],inplace=True)
        return final





   def stock_summary(self):
        stock_levels_dict,product_made_df,sales_df=self.load()
        stock_df=self.bring_together(stock_levels_dict,product_made_df,sales_df)
        print("\n",stock_df.to_string(),"\n")
      #  print(stock_df.to_string())
        return stock_df
