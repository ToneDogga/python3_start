#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:52:36 2020

@author: tonedogga
"""

import pandas as pd
import datetime as dt

logfile="IRI_reader_logfile.txt"
resultsfile="IRI_reader_results.txt"
pklsave="IRI_savenames.pkl"
colnamespklsave="IRI_savecoldetails.pkl"
fullcolnamespklsave="IRI_saveallcoldetails.pkl"
dfdictpklsave="IRI_savedfdict.pkl"
dfpklsave="IRI_fullspreadsheetsave.pkl"

wwjamsxls="IRI_ww_jams_v5.xlsx"


def save_df(df,filename):
    df.to_pickle(filename)
    return


def load_df(filename):
    return pd.read_pickle(filename)



class salestrans:
    def __init__(self):   
        self.epochs=8
    #    self.steps_per_epoch=100 
        self.no_of_batches=1000
        self.no_of_repeats=2
        
        self.dropout_rate=0.2
        self.start_point=0
        
        
        
        
column_list=list(["ww_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
        
        
               
df=pd.read_excel(wwjamsxls,-1,header=0)[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
print(df)
#df['ww_scan_week']=pd.to_datetime(df['ww_scan_week'])
df['ww_scan_week']=pd.to_datetime(df['ww_scan_week'])
df.fillna(0,inplace=True)
     #      end_date = pd.to_datetime("02/02/18") + pd.DateOffset(days=self.end_point)

print(df)
print("drop duplicates")
df.drop_duplicates(keep='first', inplace=True)
print("after drop duplicates df size=",df.shape)
#print("sort by date",df.shape[0],"records.\n")
df.sort_values(by=['ww_scan_week'], inplace=True, ascending=True)
 
#   print(df.head(5))
#   print(df.tail(5))
#print(df) 

#df["period"]=df.index.to_period('W')
#df['period'] = df['period'].astype('category')
#df['ww_scan_week'].set_index

df.set_index('ww_scan_week',inplace=True)


#newdf=df[column_list]
#print("df size=",newdf,newdf.shape,newdf.columns)
#df=df[df.columns=[bb_total_units,bb_promo_disc]]
print(df)