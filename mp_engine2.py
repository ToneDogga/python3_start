#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:25:44 2020

@author: tonedogga
"""
import subprocess as sp

#from collections import Counter
#from statistics import mean
from multiprocessing import Pool, Process, Lock, freeze_support, Queue, Lock, Manager
from os import getpid
import os
#import hashlib
import time
import pickle
import multiprocessing 
import random
import numpy as np
import pandas as pd

#from timeit import default_timer as timer


#########################################



#def main():   #classifier_mp_GA(params):   # multiprocessing stuff will go here later
tmp=sp.call('clear',shell=True)  # clear screen 'use 'clear for unix, cls for windows

 
 # start_date=pd.to_datetime("01/12/19",format="%d/%m/%y")

print("\n\nMultiprocessor test bench written by Anthony Paech 11/10/20")
print("=====================================================================")

#  sales_df=pd.read_pickle("/home/tonedogga/Documents/python_dev/sales_trans_df.pkl")
#  sales_df_augmented_savename="sales_trans_df_augmented.pkl"
all_sales_df=pd.read_pickle("/home/tonedogga/Documents/python_dev/sales_trans_df_augmented.pkl")
  
  #first_date=all_sales_df['date'].iloc[-1]
last_date=all_sales_df['date'].iloc[0]
  
 
  #print("Attache sales trans analysis.  Current save is:")
  
  
  #print("Data available:",all_sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")

  
start_date=last_date - pd.Timedelta(365, unit='d')
 
  
sales_df=all_sales_df[(all_sales_df.index>=start_date)]     # & (all_sales_df.index<=end_date)]   # & (sales_df['productgroup']==pg)].groupby(['productgroup','code'],sort=False).sum()
first_date=sales_df['date'].iloc[-1]
last_date=sales_df['date'].iloc[0]
  
#  start_date=last_date - pd.Timedelta(365, unit='d')
 
print("last 365 days..")
 

def prods_and_custs(sales_df):
     prod_list=list(set([tuple(r) for r in sales_df[['productgroup', 'product']].to_numpy()]))
     cust_list=list(set([tuple(r) for r in sales_df[['salesrep','specialpricecat', 'code']].to_numpy()]))

  #   prod_list=pd.unique(sales_df['product'])
  #   cust_list=pd.unique(sales_df['code'])
     #cust_list = cust_list[cust_list != (88.0,'OFFINV')]
  #  prod_list=list(tuple(sales_df,p) for p in prod_list)
  #   cust_list=list(tuple(sales_df,p) for p in prod_list)
 
   #  cust_list = cust_list[cust_list != (88.0,'OFFINV')]
     #print("cust_list=\n",len(cust_list))
  #   cust_list=[c for c in cust_list[2] if c!="OFFINV"]
    # prod_list=sorted(prod_list)
    # cust_list=sorted(cust_list)
     return [[c,p] for c in cust_list for p in prod_list]




def multi_function(cust_and_prod):
   # sales_df=cust_and_prod[2]
    new_df=sales_df[(sales_df['code']==cust_and_prod[0][2]) & (sales_df['product']==cust_and_prod[1][1])]  #work_in_dict['split_key']]
    if new_df.shape[0]>0:
        return [cust_and_prod[0],cust_and_prod[1],new_df]    #sales_df[(sales_df['code']==cust_and_prod[0]) & (sales_df['product']==cust_and_prod[1])]]  #work_in_dict['split_key']]
    else:
        return []


   
def main():   #classifier_mp_GA(params):   # multiprocessing stuff will go here later
   
    print("Data available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")

    cust_prod_list=prods_and_custs(sales_df)
 #   print(cust_prod_list)
    print(len(cust_prod_list),"combinations to process\n")

    start_timer = time.time()
       
    manager = multiprocessing.Manager()
    q = manager.Queue()    
   
    cpus = multiprocessing.cpu_count()
    
  
    print("Multiprocessing starting. cpus=",cpus)
 
    
 #   cust_list.insert(0,sales_df)
 #   print("cust_list=",cust_list)
    multiple_results=[]
      
    with Pool(processes=cpus) as pool:  # processes=cpus-1
       #      multiple_results.append(pool.apply_async(multi_function,args=(c,work_in_dict,q, )))  # stops, journey and poolsize, epoch length and name of q
        multiple_results.append(pool.map(multi_function,cust_prod_list))  # stops, journey and poolsize, epoch length and name of q
 
    pool.close()
   # print("pool closed.  trying to join() pool")
    pool.join()
 #   print("pool closed. join() complete")
 #   print("results=",result)
    mr=multiple_results[0]   #,cust_prod_list)]   #[0]
#    print("multiple results",mr)
    print("tidy up results.  remove empty dfs")
    nr=[elem for elem in mr if len(elem)!=0]
    with open('distribution_slices.pkl', 'wb') as f:
       pickle.dump(nr, f)
    end_timer = time.time()
    print("\nFinished. Dash total runtime:",round(end_timer - start_timer,2),"seconds.\n")

    with open('distribution_slices.pkl', 'rb') as f:
        mynewlist = pickle.load(f)
        
  #  print(mynewlist,"\n",len(mynewlist))

    print(len(cust_prod_list),"combinations processed\n")
    print("results length=",len(mr))
    
#    print(nr,"\n Final result list length=",len(nr))

    print("\n Final result list length=",len(mynewlist))
    
    print("\n")
    print(mynewlist[:12])

    return



if __name__ == '__main__':
    main()


  