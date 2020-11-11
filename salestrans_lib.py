#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:41:08 2020

@author: tonedogga
"""
# =============================================================================
#   sales trans class
# 
#   load from excel into a pandas df
# 
#   display_df print a df
#
##         #   query of AND's - input a list of tuples.  ["&",(field_name1,value1) and (field_name2,value2) and ...]
# #            return a slice of the df as a copy
# # 
# #        a query of OR 's  -  input a list of tuples.  ["|",(field_name1,value1) or (field_name2,value2) or ...]
# #            return a slice of the df as a copy
# #
# #        a query_between is only a triple tuple  ["between",(fieldname,startvalue,endvalue)]
# # 
# #        a query_not is only a single triple tuple ["!",(fieldname,value)]   
#
#   encode_query_name - take a query list and turn it into a name that can be used as a .pkl filename
#
#   decode_query_name - take a query pkl filename and return a list as an underlying query list
# 
#   save_df - pandas pickle df to (".pkl") using the encoded query name
#
#   load_df - pandas unpickle from (".pkl") using the decoded query name
#
#
# =============================================================================


import numpy as np
import pandas as pd

from p_tqdm import p_map
from datetime import datetime
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

import time
import joblib

import os

import pickle
import codecs

from pathlib import Path

import query_dict as qd

class salestrans_df:
    def __init__(self,outd,rtld):  #,filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]):   #, m=[["L","R","-","T"],["T","-","L","R"],["R","L","T","-"],["-","T","R","L"]]):
       self.output_dir = outd  #self.log_dir("salestrans_outputs")
       self.rootlogdir= rtld
    #     os.makedirs(self.output_dir, exist_ok=True)


        #     self.f=0
    #     self.t="test2"
    #   #  print("filenames=",self.filenames)
    #     return


    
    def load_excel(self,filename):
        print("load:",filename)
        new_df=pd.read_excel(filename,sheet_name="AttacheBI_sales_trans",usecols=range(0,17),verbose=False)  # -1 means all rows  
        new_df['date'] = pd.to_datetime(new_df.date)  #, format='%d%m%Y')
        return new_df

    
    
    def load(self,filenames,renew):  # filenames is a list of xlsx files to load and sort by date
        if renew:
 #           print("Loading from excel:",filenames,"\nload:",filenames[0])
            print("Loading from excel:",filenames,"\nload:",filenames)

           # df=pd.read_excel(filenames[0],sheet_name="AttacheBI_sales_trans",usecols=range(0,17),verbose=False)  # -1 means all rows   
            df=pd.DataFrame([])
            #   price_df=pd.read_excel("salestrans.xlsx",sheet_name="prices",usecols=range(0,dd.price_width),header=0,skiprows=[0,2,3],index_col=0,verbose=False)  # -1 means all rows  
         #   price_df=price_df.iloc[:-2]
         #   price_df = price_df.rename_axis("product")
         
  #          df=df.append(p_map(self.load_excel,filenames[1:])) 
            df=df.append(p_map(self.load_excel,filenames)) 
             
            df.fillna(0,inplace=True)
            df=df[(df.date.isnull()==False)]
            
            print("drop duplicates")
            df.drop_duplicates(keep='first', inplace=True)
            print("after drop duplicates df size=",df.shape)
            print("sort by date",df.shape[0],"records.\n")
            df.sort_values(by=['date'], inplace=True, ascending=False)
            
            df["period"]=df.date.dt.to_period('D')
         
            df['period'] = df['period'].astype('category')
            df.set_index('date',inplace=True,drop=False) 
            df=df.rename(columns=qd.rename_columns_dict)  
            self.save_query(df,qd.queries['all'],root=True)
            #df.to_pickle("./st_df.pkl",protocol=-1)
        else:
        #    query_handle=self.encode_query_name([])[:-1]
         #   my_file = Path("./"+str(query_handle)+".pkl")
          #  if my_file.is_file():
           #     df=pd.read_pickle(my_file)
            df=self.load_query(qd.queries['all'],root=True,fileinputtype=False)
          #  else:
          #      print("no pickle created yet",my_file)
          #      df=pd.DataFrame([])
   #     if df.shape[0]>0:
   #         df=df.rename(columns=qd.rename_columns_dict)  
        return df   #.rename(columns=qd.rename_columns_dict,inplace=True)
    
      
    
    # def log_dir(self,prefix=""):
    #     now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    #     root_logdir = "./salestrans_outputs"
    #     if prefix:
    #         prefix += "-"
    #     name = prefix + "run-" + now
    #     return "{}/{}/".format(root_logdir, name)
    


      


   # def display_df(self,df):
   #     print("display_df=\n",df)
        
        
    
    def encode_query_name(self,query_name):
        return codecs.encode(pickle.dumps(query_name), "base64").decode()


    def decode_query_name(self,filename):
        return pickle.loads(codecs.decode(filename.encode(), "base64")) #[:-1]


   
    def save_query(self,df,query,root):
     #   print("save query",query_name)
      #  if df.shape[0]>0:
        filename=self.encode_query_name(query)[:-1]  
        if len(filename)>249:
            filename=filename[:250]
      
    #    print("load query name=",query_handle,len(query_handle),"filename=",filename,len(filename))

     #   print("filename length=",len(filename))
    #    if len(ffilename)>249:
    #        filename=ffilename[:250]
    #    else:
    #        filename=ffilename
    #    print("save query",query,len(query)," filename",filename,len(filename))
        if root: 
            df.to_pickle(self.rootlogdir+filename+".pkl",protocol=-1)
        else:     
            df.to_pickle(self.output_dir+filename+".pkl",protocol=-1)
        return filename
      #  else:
      #      return "empty"
    
    
       
    def load_query(self,query,root,fileinputtype):
       # filename=Path("./"+filename)   
   #     print("load query",filename)
        if fileinputtype:
            filename=query
        else:    
            filename=self.encode_query_name(query)[:-1]  

 #       filename=self.decode_query_name(query)[:-1]  
        if len(filename)>249:
            filename=filename[:250]
    
    #    print("load query name=",query,len(query),"filename=",filename,len(filename))
     
      #      query_name=qd.queries
      #  else:    
            
    #    print("load query",query_name,"filename",filename)
        if root:
            df=pd.read_pickle(self.rootlogdir+filename+".pkl")            
        else:    
            df=pd.read_pickle(self.output_dir+filename+".pkl")
        return df

    

    def query_df(self,df,query_name):
# =============================================================================
#         
#         #   query of AND's - input a list of tuples.  ["AND",(field_name1,value1) and (field_name2,value2) and ...]
#             the first element is the type of query  -"&"-AND, "|"-OR, "!"-NOT, "B"-between
# #            return a slice of the df as a copy
# # 
# #        a query of OR 's  -  input a list of tuples.  ["OR",(field_name1,value1) or (field_name2,value2) or ...]
# #            return a slice of the df as a copy
# #
# #        a query_between is only a triple tuple  ["B",(fieldname,startvalue,endvalue)]
# # 
# #        a query_not is only a single triple tuple ["NOT",(fieldname,value)]   
# 
#         
# =========================================================================
       if (query_name==[]) | (df.shape[0]==0):
           return df.copy(deep=True) 
       else :   
           if (query_name[0]=="AND") | (query_name[0]=='OR') | (query_name[0]=="B") | (query_name[0]=="NOT"):
                operator=query_name[0]
              #  print("valid operator",operator)
                query_list=query_name[1:]
             #   print("quwery",query_list)
                new_df=df.copy()
                if operator=="AND":
                    for q in query_list:    
                        new_df=new_df[(new_df[q[0]]==q[1])].copy(deep=True) 
                    #    print("AND query=",q,"&",new_df.shape) 
                 #   print("new_df=\n",new_df)    
                elif operator=="OR":
                    new_df_list=[]
                    for q in query_list:    
                        new_df_list.append(new_df[(new_df[q[0]]==q[1])].copy(deep=True)) 
                     #   print("OR query=",q,"|",new_df_list[-1].shape)
                    new_df=new_df_list[0]    
                    for i in range(1,len(query_list)):    
                        new_df=pd.concat((new_df,new_df_list[i]),axis=0)   
                  #  print("before drop",new_df.shape)    
                    new_df.drop_duplicates(keep="first",inplace=True)   
                  #  print("after drop",new_df.shape)
                elif operator=="NOT":
                    for q in query_list:    
                        new_df=new_df[(new_df[q[0]]!=q[1])].copy(deep=True) 
                   #     print("NOT query=",q,"NOT",new_df.shape)  
                   
                  #   new_df_list=[]
                  #   for q in query_list:    
                  #       new_df_list.append(new_df[(new_df[q[0]]!=q[1])].copy()) 
                  #    #   print("OR query=",q,"|",new_df_list[-1].shape)
                  #   new_df=new_df_list[0]    
                  #   for i in range(1,len(query_list)):    
                  #       new_df=pd.concat((new_df,new_df_list[i]),axis=0)   
                  # #  print("before drop",new_df.shape)    
                  #   new_df.drop_duplicates(keep="first",inplace=True)   
    
                   
                elif operator=="B":
                  #  if (len(query_list[0])==3):
                    for q in query_list:
                    #        print("between ql=",q[1],q[2])
                        start=q[1]
                        end=q[2]
                        new_df=new_df[(new_df[q[0]]>=q[1]) & (new_df[q[0]]<=q[2])].copy(deep=True) 
                     #       print("Beeterm AND query=",q,"&",new_df.shape) 
                   # else:
                   #     print("Error in between statement")
                
                else:
                    print("operator not found\n")
                
                
                return new_df.copy(deep=True)
                      
           else:
                print("invalid operator")
                return pd.DataFrame([])
    
    
            
  