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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


#import matplotlib.pyplot as plt

import timeit

from collections import Counter,OrderedDict
 


def save_model(model,filename):
    #filename = 'finalized_model.sav'
    #    joblib.dump(regressor,open("SGDRegressorNS.p","wb"))

    joblib.dump(model, filename)
    return 

def load_model(filename):
    # some time later...

    # load the model from disk

    loaded_model = joblib.load(filename)
    return loaded_model


def main():


    f=open(cfg.outfile_predict,"w")

    print("\n\nData Regression sales prediction tool - By Anthony Paech 5/12/19\n")
#    print("Loading",sys.argv[1],"into pandas for processing.....")
    print("Loading",cfg.infilename,"into pandas for processing.....")

    print("Results Reported to",cfg.outfile_predict)

    f.write("\n\nData Regression sales prediction tool - By Anthony Paech 5/12/19\n\n")
 #   f.write("Loading "+sys.argv[1]+" into pandas for processing.....\n")
    f.write("Loading "+cfg.infilename+" into pandas for processing.....\n")

    f.write("Results Reported to "+cfg.outfile_predict+"\n")

##    df=read_excel(sys.argv[1],-1)  # -1 means all rows
##    if df.empty:
##        print(sys.argv[1],"Not found.")
##        sys.exit()




######################################################3

  #  start_d=input("Start date for predictions? YYYY/MM/DD  (you can only predict 1 year in advance of current date):")
    start_d="2019/11/01"   #str(dt.datetime.now)   #input("Start date for predictions? YYYY/MM/DD  (you can only predict 1 year in advance of current date):")

    start_date=pd.to_datetime(start_d)
    one_year_ago = (start_date+relativedelta(years=-1)).strftime('%Y/%m/%d')
 
    #one_year_ago = (dt.datetime.now()+relativedelta(years=-1)).strftime('%Y/%m/%d')
    print("One year back from start date:",one_year_ago,"\n\n")

   # dbd=pd.DataFrame()
    
    dbd=pd.read_csv(cfg.datasetpluspredict,header=0)

    dbd.drop(columns=["qty","predict_qty"],inplace=True)

    dbd.sort_values(by=["date"],axis=0,ascending=[True],inplace=True)

   # print(df.head(10))
    print("data import shape:",dbd.shape)
    f.write("data import shape:"+str(dbd.shape)+"\n")




    print("Prepare data....")
    dbd.dropna(inplace=True)
    #df['Date']= pd.to_datetime(df['Date'])
    dbd['date'] = pd.to_datetime(dbd['date'])    #pd.date().map(dt.datetime.toordinal).astype(int)
    dbd['date_encode'] = dbd['date'].map(dt.datetime.toordinal).astype(int)

   # dbd['day_delta'] = (dbd.date-dbd.date.min()).dt.days.astype(int)
    dbd.drop(columns=["date"],inplace=True)

    #  encode "code", "product"
    
    label_encoder=LabelEncoder()
    dbd["prod_encode"] = label_encoder.fit_transform(dbd["product"].to_numpy())
    joblib.dump(label_encoder,open(cfg.product_encode_save,"wb"))
    dbd.drop(columns=["product"],inplace=True)

    label_encoder=LabelEncoder()
    dbd["code_encode"] = label_encoder.fit_transform(dbd["code"].to_numpy())
    joblib.dump(label_encoder,open(cfg.code_encode_save,"wb"))
    dbd.drop(columns=["code"],inplace=True)

   # dbd.drop(columns=["day_delta"],inplace=True)

    #print("dbd=\n",dbd)



##########################################################
# create predictions


    regressor_best=joblib.load(open(cfg.RFR_save,"rb"))
    print("RFR model loaded from:",cfg.RFR_save)
    f.write("RFR model loaded from:"+str(cfg.RFR_save)+"\n")
#    print(dbd)
  #  Xr_df_cols=Xr_df[:,:4]
#    print(dbd.columns)
    predictions = regressor_best.predict(dbd)

    #j=pd.DataFrame(dbd,columns=["code_encode","prod_encode","date_encode","day_order_delta"])
    encoder=joblib.load(open(cfg.code_encode_save,"rb"))
    dbd["code"]=encoder.inverse_transform(dbd["code_encode"].astype(int).to_numpy())
    dbd.drop(columns=["code_encode"],inplace=True)


    encoder=joblib.load(open(cfg.product_encode_save,"rb"))
    dbd["product"]=encoder.inverse_transform(dbd["prod_encode"].astype(int).to_numpy())
    dbd.drop(columns=["prod_encode"],inplace=True)

    dbd["date"] = dbd.date_encode.astype(int).map(dt.datetime.fromordinal)
    dbd.drop(columns=["date_encode"],inplace=True)

   # dbd["qty"]=fulldataset["qty"]

    dbd["predict_qty"]=predictions.reshape(-1,1)
   # dbd2=pd.DataFrame(np.hstack((dbd.to_numpy(),predictions.reshape(-1,1))),columns=["day_order_delta","code","product","date","predict_qty"])



################################################################################################    

    last_year=(dbd["date"]>=one_year_ago)    # & (df2.day_order_delta>=4) & (df2.day_order_delta<=45))

    dbd2=dbd.loc[last_year,:]

    #dbd2=dbd[dbd["date">=one_year_ago]].index
    #print(dbd2)
    print(dbd2.columns)

  #  print("j=\n",j)


#####################################################################################




    
  #  now = dt.datetime.now()   #.strftime('%d/%m/%Y')
 #   now_dt = datetime.strptime(now, '%Y/%m/%d').date()
  #  print("now=",now)
    dbd2["new_date"]=start_date  #.dt.date
    dbd2['new_date'] = dbd2['new_date'] + pd.to_timedelta(dbd2['day_order_delta'], unit='d')
    if cfg.dateformat=="year/week":
        dbd2['predict_date']=dbd2['new_date'].dt.strftime('%Y/%w')    # ('%Y/%m/%d")
    elif cfg.dateformat=="year/month/day":
        dbd2['predict_date']=dbd2['new_date'].dt.strftime('%Y/%m/%d')
    elif cfg.dateformat=="year/month":
        dbd2['predict_date']=dbd2['new_date'].dt.strftime('%Y/%m')
        
   # new_datetime_obj = datetime.strptime(orig_datetime_obj.strftime('%d-%m-%y'), '%d-%m-%y').date()
 
    dbd2= dbd2[["code","product","predict_date","predict_qty"]]
    
    dbd2.sort_values(by=["predict_date"],axis=0,ascending=[False],inplace=True)
    dbd2["predict_qty"]=dbd2["predict_qty"].astype(float).round(0)    #{"predict_qty" : 1})
    dbd2["predict_qty_ctnsof8"]=(dbd2["predict_qty"]/8).astype(float).round(0)
    #   print(k)
    #print("Sales Qty Predictions=\n",dbd2[0:100].to_string())
    
   
#############################################################################3
    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

    table = pd.pivot_table(dbd2, values='predict_qty', index=['product', 'predict_date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)   #, observed=True)
  #  print("\ntable=\n",table.head(5))
    f.write("\n\n"+table.to_string())

    table2 = pd.pivot_table(dbd2, values='predict_qty', index=['code', 'predict_date'],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable2=\n",table2.head(5))
    f.write("\n\n"+table2.to_string())

    table3 = pd.pivot_table(dbd2, values='predict_qty', index=['predict_date',"code"],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable3=\n",table3.head(5))
    f.write("\n\n"+table3.to_string())

    table4 = pd.pivot_table(dbd2, values='predict_qty', index=['predict_date',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable4=\n",table4.head(5))
    f.write("\n\n"+table4.to_string())

    table5 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['product', 'predict_date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable5=\n",table5.head(5))
    f.write("\n\n"+table5.to_string())

    table6 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['code', 'predict_date'],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable6=\n",table6.head(5))
    f.write("\n\n"+table6.to_string())

    table7 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['predict_date',"code"],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable7=\n",table7.head(5))
    f.write("\n\n"+table7.to_string())

    table8 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['predict_date',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable8=\n",table8.head(5))
    f.write("\n\n"+table8.to_string())


    with pd.ExcelWriter(cfg.outxlsfile) as writer:  # mode="a" for append
        table.to_excel(writer,sheet_name="Units1")
        table2.to_excel(writer,sheet_name="Units2")
        table3.to_excel(writer,sheet_name="Units3")
        table4.to_excel(writer,sheet_name="Units4")
        table5.to_excel(writer,sheet_name="CtnsOfEight5")
        table6.to_excel(writer,sheet_name="CtnsOfEight6")
        table7.to_excel(writer,sheet_name="CtnsOfEight7")
        table8.to_excel(writer,sheet_name="CtnsOfEight8")

    print("\n\nSales Prediction results from",start_d,"written to spreadsheet",cfg.outxlsfile,"\n\n")
    f.write("\n\nSales Prediction results from "+str(start_d)+" written to spreadsheet:"+str(cfg.outxlsfile)+"\n\n")
    
   
##############################################################################


    f.close()
    return

    

if __name__ == '__main__':
    main()
