#!/usr/bin/python3
from __future__ import print_function
from __future__ import division


import numpy as np
import pandas as pd
import csv
import sys
import datetime as dt
import joblib
import pickle

import sales_regression_cfg as cfg

from dateutil.relativedelta import relativedelta


##import matplotlib.pyplot as plt
##
##import timeit
##
##from collections import Counter,OrderedDict
##    
##from sklearn.metrics import roc_auc_score
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import confusion_matrix
##from sklearn.metrics import classification_report
##from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
##
###import gc
##
##from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, MinMaxScaler, minmax_scale

##from sklearn.feature_extraction import DictVectorizer
##from sklearn.linear_model import SGDRegressor
##from sklearn.svm import SVR
##from sklearn.svm import SVC
##from sklearn.svm import LinearSVR
##from sklearn.svm import LinearSVC
##
##   
##from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
##from sklearn.model_selection import GridSearchCV
##from sklearn.model_selection import cross_val_score
##from sklearn.model_selection import learning_curve
##from sklearn.model_selection import ShuffleSplit
##
##
##from sklearn.tree import DecisionTreeRegressor
##from sklearn.ensemble import RandomForestRegressor
##from sklearn.ensemble import RandomForestClassifier
##from sklearn.linear_model import SGDClassifier
##
##
##from sklearn.pipeline import Pipeline
##
##from sklearn.utils.testing import ignore_warnings
##from sklearn.exceptions import ConvergenceWarning  





##def column_scale(s):
##
##    length=len(np.trim_zeros(s.sum(axis=0),'b'))
##    sclean=np.delete(s,np.s_[length::],axis=1)
##    scaled=sclean*length
##    
##    ysum=sclean.sum(axis=1)
##    xsum=sclean.sum(axis=0)
##
##    # now scale the range from 0 to 1
##    min_max_scaler = MinMaxScaler()  
##    final_scaled = min_max_scaler.fit_transform(scaled)+0.5
##    # final result is between 0.5  and 1.5
##    print("final scaled mean=",final_scaled.mean(),"median=",np.median(final_scaled,axis=0),"min=",final_scaled.min(),"max",final_scaled.max())
##    correction_factor=1/final_scaled.mean()
##    #correction_factor=1/np.median(final_scaled,axis=0)
##
##    print("mean based correction_factor=",correction_factor)
##  #  print("\nfinal_scaled=",final_scaled,"\nscaled",scaled,"\nscaled shape=",scaled.shape,"\nlength=",length,"\nysum=",ysum,"\nxsum=",xsum)
##    return final_scaled



def smoothing_distributions(df,product_list,bins):
    #  create an array of prod_encodes as keys and the values are a counter is the day_delta as a key and the value is the sum of the qty sales for each day
#    e=df.groupby(["prod_encode",pd.qcut(df.day_delta,q=noofbins,labels=range(noofbins))],as_index=[False,False])[['qty']].agg('sum').fillna(0)
  #  e=df.groupby(["prod_encode",pd.cut(df.day_delta,bins=bins,labels=range(len(bins)-1))],as_index=[False,False])[['qty']].agg('sum').fillna(0)
    e=df.groupby(["product",pd.cut(df.day_delta,bins=bins,labels=range(len(bins)-1))],as_index=[False,False])[['qty']].agg('sum').fillna(0)

    f=e.unstack(fill_value=0.0)
    print("smoothing shape=",f.shape)
##    print("unstack array saved to","unstack.xlsx")   #cfg.scalerdump1)
 #   f.to_excel("unstack.xlsx")   #, index=False)

    active_bins=len(f.columns)
    print("active bins=",active_bins)

    sumf=f.qty.sum(axis=1).to_numpy()    #.reshape(-1,1)     #.tolist()
    i=0
    smoothing=[]   #np.empty((1,1),dtype=float)   #empty((len(prod_list),noofbins))    #np.zeros((len(prod_list,noofbins),dtype=float)
  #  print("smoothing shpae",smoothing.shape)
    for prod in product_list:
        row=f.iloc[i,:].to_numpy()
        brkdwn=row/sumf[i]   #.reshape(-1,1)
        smoothing.append(brkdwn)
        i+=1
    r=np.asarray(smoothing)
    s=np.clip(r,0,1)
  #  summean=s.sum(axis=0).mean()
   # print("before adjusted=sum and mean",s.sum(axis=0).mean())

   # adjust=1/summean
   # s*=adjust
    smean=s.sum(axis=0).mean()

    print("adjusted=sum and mean",smean)
    sdf = pd.DataFrame(s)
    sdf.insert(0,"product",product_list)
   #

   
#    scaled=column_scale(s)  # for each row which is a product code list, multiply by the number of active bins  and then scale between 0.5 and 1.5

   
   # print("scaled s",s[0:100])
   # print(s.shape,s.min(),s.max(),s.mean(),s.sum(axis=1))
   # write scaler to excel
 #   sdf = pd.DataFrame (scaled)
   # sdf.insert(0,"prod_encode",prod_list)


#### save to xlsx file
##    print("Scaler array saved to",cfg.scalerdump)


    return sdf,smean


def main():
    big_trans_file="NAT-raw.xlsx"
    print("Create a product sales distribution breakdown file from",big_trans_file)
    endloop=False
    while not endloop:
        answer=str(input("Overwrite current product sales distribution breakdown file? (y/n)"))
        endloop=(answer=="y" or answer=="n")
    if answer=="y":
        print("Importing",big_trans_file,"into pandas.")
        dfx=pd.read_excel(big_trans_file)   #,cfg.importrows)  # -1 means all rows
        if dfx.empty:
            print(cfg.infilename,"Not found. Check sales_regression_cfg.py file")
            sys.exit()
        df=dfx.iloc[:,0:17]
        
      
        
       # del df  # clear memory 

        print("Imported into pandas=\n",df.columns,"\n",df.shape)    #head(10))


        
    ##################################################33
       # Remove extranous fields
        print("Prepare data....")
        b4=df.shape[1]
        print("Remove columns not needed.")
    #    df.drop(columns=["cat","code","costval","glset","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"],inplace=True)
        df.drop(columns=cfg.excludecols,inplace=True)

        print("Columns deleted:",b4-df.shape[1]) 

        print("Delete rows for productgroup <10 or >15.")
        b4=df.shape[0]
        df.drop(df[(df["productgroup"]<10) | (df["productgroup"]>15)].index,inplace=True)
        print("Rows deleted:",b4-df.shape[0])
        


        df.dropna(inplace=True)


    #############################################
        # encode

        df['date_encode'] = df['date'].map(dt.datetime.toordinal).astype(int)
        df['day_delta'] = (df.date-df.date.min()).dt.days.astype(int)
        

        label_encoder=LabelEncoder()
        df["code_encode"] = label_encoder.fit_transform(df["code"].to_numpy())
        joblib.dump(label_encoder,open(cfg.code_encode_save,"wb"))
        df.drop(columns=["code"],inplace=True)
        print(df.columns)


        df2=df.sort_values(by=["code_encode","product","day_delta"],ascending=[True,True,False])   #,inplace=True)

        df2["day_order_delta"]=round(df2.day_delta.diff(periods=-1),0)
       
        cust_list=list(set(df2["code_encode"].tolist()))
     #   prod_list=list(set(df2["prod_encode"].tolist()))
        product_list=list(set(df2["product"].tolist()))

        cust_list_len=len(cust_list)
        product_list_len=len(product_list)

        scaler_df,smean =smoothing_distributions(df2,product_list,cfg.bins)
        print("scaler mean=",smean)

        print("Saving product sales distribution breakdown file to",cfg.scalerdump)
      #  scaler_df = pd.DataFrame(scaled)
        print(scaler_df.columns)
     #   scaler_df.to_excel("sarray.xlsx", index=False)
        scaler_df.to_excel(cfg.scalerdump, index=False)
 
 
    return

    

if __name__ == '__main__':
    main()

