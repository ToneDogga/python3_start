#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 09:43:19 2020

@author: tonedogga
"""
import pandas as pd
import datetime as dt

smoothing_mat=4
sales_trans_filenames=["allsalestrans2018.xlsx","allsalestrans2020.xlsx","salestrans.xlsx"]   #"allsalestrans190520.xlsx",

rename_columns_dict={'specialpricecat':'spc','productgroup':'pg'}
#
queries={
    'all':[[]],
    # "pasa ts":[["OR",("product","TS300"),("code","FLPAS"),("salesrep","36")]],
    # 'pasa ts2':[["AND",("product","TS300"),("code","FLPAS"),("salesrep","36")]],
    # 'cnot pasa ts2':[['NOT',("product","TS300"),("code","FLPAS"),("salesrep","36")]],
    # 'ctns':[["B",("qty",8,16)]],
    # 'between dates':[["B",("date",pd.to_datetime("2019-01-01"),pd.to_datetime("2019-12-31"))]],
    # 'between dates2':[["B",("date",pd.to_datetime("2018-01-01"),pd.to_datetime("2018-12-31")+pd.offsets.Day(7))]],   
    # 'last 365':[["B",("date",pd.to_datetime("2020-01-01")+pd.offsets.Day(-365),pd.to_datetime("2020-12-31")+pd.offsets.Day(7))]],   
    # 'last 365 test':[["B",("date",pd.to_datetime("today")+pd.offsets.Day(-365),pd.to_datetime("today"))]],   
    # "not pasa ts2":[["NOT",("product","TS300"),("code","FLPAS")]],
    # "pasa ts3":[["OR",("product","TS300"),("code","FLPAS"),("salesrep","36")]],
    # "pasa ts4":[["OR",("product","TS300"),("code","FLFRE"),("salesrep","36")]],
    # "pasa ts5":[["OR",("product","SJ300"),("code","FLFRE"),("salesrep","36")]],
    # "pasa ts6":[["OR",("product","RJ300"),("code","FLPAS"),("salesrep","36")]],
    # "pasa ts7":[["OR",("product","TS300"),("code","FLBRI"),("salesrep","36")]],
    # "pasa ts8":[["AND",("product","TS300"),("code","FLBRI"),("salesrep","36")]],
    # "pasa ts9":[["AND",("product","TS300"),("code","FLBRI"),("salesrep","36")],["B",("qty",8,32)]], #,["B",("date",pd.to_datetime("today"),pd.to_datetime("today"))]]
    # 'between dates3':[['NOT',('code','CASHSHOP')],["B",("date",pd.to_datetime("2020-01-03"),pd.to_datetime("2020-01-03"))]],
    # "not shop":[["NOT",('specialpricecat',92)]],
    # "online":[["AND",('glset','ONL')],["B",("date",pd.to_datetime("2020-01-03"),pd.to_datetime("2020-01-03"))]]
    "harris farm TY all":[["AND",("spc",122)],["B",("date",pd.to_datetime("today")+pd.offsets.Day(-365),pd.to_datetime("today"))]],
    "harris farm TY mealbases andf jams":[["AND",("spc",122)],["OR",('pg','14'),("pg","10"),('pg','11')],["B",("date",pd.to_datetime("today")+pd.offsets.Day(-365),pd.to_datetime("today"))]],
    "harris farm all":[["AND",("spc",122)]],
    "harris farm mealbases":[["AND",("spc",122),('pg','14')]]
 
 #   "not shop or majors":[["NOT",('spc',92),('spc',10),('spc',12)]]
 #   "not shop or majors last 365 days":[["NOT",('glset',"SHP"),('spc',10),('spc',12)],["B",("date",pd.to_datetime("today")+pd.offsets.Day(-365),pd.to_datetime("today"))]]

  #  "not shop or majors2":[["NOT",('specialpricecat',92)],['NOT',('specialpricecat',10)],['NOT',('specialpricecat',12)]]

     }   