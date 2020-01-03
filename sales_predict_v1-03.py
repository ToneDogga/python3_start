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

    print("\n\nData Regression sales prediction tool - By Anthony Paech 12/12/19\n")
#    print("Loading",sys.argv[1],"into pandas for processing.....")
    print("Loading",cfg.datasetworking,"into pandas for processing.....")

    print("Results Reported to",cfg.outfile_predict)

    f.write("\n\nData Regression sales prediction tool - By Anthony Paech 5/12/19\n\n")
 #   f.write("Loading "+sys.argv[1]+" into pandas for processing.....\n")
    f.write("Loading "+cfg.datasetworking+" into pandas for processing.....\n")

    f.write("Results Reported to "+cfg.outfile_predict+"\n")

##    df=read_excel(sys.argv[1],-1)  # -1 means all rows
##    if df.empty:
##        print(sys.argv[1],"Not found.")
##        sys.exit()




######################################################3


   # dbd=pd.DataFrame()
    
    dbd=pd.read_csv(cfg.datasetworking,header=0)

    dbd.drop(columns=["product","date","last_order_upspd","prod_scaler","day_delta"],inplace=True)
 

    binmask=(dbd["bin_no"]>=cfg.startbin)
    dbd2=dbd[binmask].copy(deep=True)
    
    dbd2.sort_values(by=["bin_no","code_encode","prod_encode"],axis=0,ascending=[True,True,True],inplace=True)

   # print(df.head(10))
    print("data import shape:",dbd2.shape)
    f.write("data import shape:"+str(dbd2.shape)+"\n")




    print("Prepare data....")
    dbd2.dropna(inplace=True)



    #df['Date']= pd.to_datetime(df['Date'])
   # dbd['date'] = pd.to_datetime(dbd['date'])    #pd.date().map(dt.datetime.toordinal).astype(int)
   # dbd['date_encode'] = dbd['date'].map(dt.datetime.toordinal).astype(int)

   # dbd['day_delta'] = (dbd.date-dbd.date.min()).dt.days.astype(int)
   # dbd.drop(columns=["date"],inplace=True)

    #  encode "code", "product"
    
##    label_encoder=LabelEncoder()
##    dbd["prod_encode"] = label_encoder.fit_transform(dbd["product"].to_numpy())
##    joblib.dump(label_encoder,open(cfg.product_encode_save,"wb"))
##    dbd.drop(columns=["product"],inplace=True)
##
##    label_encoder=LabelEncoder()
##    dbd["code_encode"] = label_encoder.fit_transform(dbd["code"].to_numpy())
##    joblib.dump(label_encoder,open(cfg.code_encode_save,"wb"))
##    dbd.drop(columns=["code"],inplace=True)
##
#    dbd.drop(columns=["day_delta","productgroup"],inplace=True)
##
##   # print("dbd=\n",dbd)
##
    qtyarray=dbd2.qty.to_numpy()
    binarray=dbd2.bin_no.to_numpy()
    prodgrouparray=dbd2.productgroup.to_numpy()
##    print("qtyarray.shape=",qtyarray.shape,"\n",qtyarray)
## 
    dbd2.drop(columns=["qty","bin_no","productgroup"],inplace=True)
    print(dbd2.columns)


##########################################################
# create predictions


    regressor_best=joblib.load(open(cfg.RFR_save,"rb"))
    print("RFR model loaded from:",cfg.RFR_save)
    f.write("RFR model loaded from:"+str(cfg.RFR_save)+"\n")
#    print(dbd)
  #  Xr_df_cols=Xr_df[:,:4]
#    print(dbd.columns)
    predictions = regressor_best.predict(dbd2)

    dbd2["predict_qty"]=predictions.reshape(-1,1).astype(float).round(0)
   # dbd2=pd.DataFrame(np.hstack((dbd.to_numpy(),predictions.reshape(-1,1))),columns=["day_order_delta","code","product","date","predict_qty"])



########################################################


    dbd2["qty"]=qtyarray.tolist()
    dbd2["bin_no"]=binarray.tolist()
    dbd2["productgroup"]=prodgrouparray.tolist()


    #j=pd.DataFrame(dbd,columns=["code_encode","prod_encode","date_encode","day_order_delta"])
    encoder=joblib.load(open(cfg.code_encode_save,"rb"))
    dbd2["code"]=encoder.inverse_transform(dbd2["code_encode"].astype(int).to_numpy())
    dbd2.drop(columns=["code_encode"],inplace=True)


    encoder=joblib.load(open(cfg.product_encode_save,"rb"))
    dbd2["product"]=encoder.inverse_transform(dbd2["prod_encode"].astype(int).to_numpy())
    dbd2.drop(columns=["prod_encode"],inplace=True)

#    dbd2["date"] = dbd2.date_encode.astype(int).map(dt.datetime.fromordinal)


    dbd2["date"]=(pd.to_datetime("2/2/20")+pd.to_timedelta((dbd2["bin_no"]-cfg.startbin)*7,unit="d")).dt.strftime('%Y/%m/%d')

    dbd2.drop(columns=["date_encode","bin_no","scaled_upspd","day_order_delta"],inplace=True)

 
    dbd2= dbd2[["date","code","productgroup","product","qty","predict_qty"]]
  #  dbd2= dbd2[["date","code","product","predict_qty"]]
     
    dbd2.sort_values(by=["date","code","productgroup","product"],axis=0,ascending=[True,True,True,True],inplace=True)
   # dbd2["predict_qty"]=dbd2["predict_qty"].astype(float).round(0)    #{"predict_qty" : 1})
    dbd2["predict_qty_ctnsof8"]=(dbd2["predict_qty"]/8).astype(float).round(0)
    #   print(k)
    #print("Sales Qty Predictions=\n",dbd2[0:100].to_string())


    print(dbd2)
    print(dbd2.columns)

##################################3
    # extend dbd2 a further 1 year


    df_pp=pd.read_csv(cfg.datasetpluspredict,header=0)

    df_pp.drop(columns=["GMV","day_order_delta","scaled_upspd"],inplace=True)

    df_pp["date"]=pd.to_datetime(df_pp["date"],format="%Y/%m/%d")   #.dt.strftime('%Y/%m/%d')
    df_pp.sort_values(by=['date'],ascending=True,inplace=True)
    df_pp["predict_qty_ctnsof8"]=(df_pp["predict_qty"]/8).astype(float).round(0)

    df_pp= df_pp[["date","code","productgroup","product","qty","predict_qty","predict_qty_ctnsof8"]]

    latest_date=pd.to_datetime(df_pp.date.max())
  #  print("latest date=",latest_date)

    first_date=pd.to_datetime(pd.to_datetime(latest_date)+pd.to_timedelta(-1,unit="Y"))   #.dt.strftime('%Y/%m/%d')

    print(type(first_date)) 
    
    print("first_date=",first_date,"latest date=",latest_date)
    mask=((df_pp["date"]>=first_date) & (df_pp["date"]<=latest_date))
    extendyear=df_pp.loc[mask].copy(deep=True)
    extendyear["date"]=pd.to_datetime(pd.to_datetime(extendyear.date)+pd.to_timedelta(2,unit="Y"),format="%Y/%m/%d") 
    extendyear.sort_values(by=['date'],ascending=True,inplace=True)
    extendyear["date"]=extendyear["date"].dt.strftime("%Y/%m/%d")
    extendyear["predict_qty"]=extendyear["predict_qty"].round(0)   
    extendyear["predict_qty_ctnsof8"]=extendyear["predict_qty_ctnsof8"].round(0)   

    #print("extendyear=\n",extendyear)            

    dbd2=pd.concat((dbd2,extendyear))
    
    print("new dbd2=\n",dbd2)

    dbd2.to_excel("dbd2.xlsx")
    
   
#############################################################################3
    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

    table = pd.pivot_table(dbd2, values='predict_qty', index=['productgroup','product', 'date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)   #, observed=True)
  #  print("\ntable=\n",table.head(5))
    f.write("\n\n"+table.to_string())

    table2 = pd.pivot_table(dbd2, values='predict_qty', index=['code', 'date'],columns=['productgroup','product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable2=\n",table2.head(5))
    f.write("\n\n"+table2.to_string())

    table3 = pd.pivot_table(dbd2, values='predict_qty', index=['date',"code"],columns=['productgroup','product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable3=\n",table3.head(5))
    f.write("\n\n"+table3.to_string())

    table4 = pd.pivot_table(dbd2, values='predict_qty', index=['date','productgroup',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable4=\n",table4.head(5))
    f.write("\n\n"+table4.to_string())

    table5 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['productgroup','product', 'date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable5=\n",table5.head(5))
    f.write("\n\n"+table5.to_string())

    table6 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['code', 'date'],columns=['productgroup','product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable6=\n",table6.head(5))
    f.write("\n\n"+table6.to_string())

    table7 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['date',"code"],columns=['productgroup','product'], aggfunc=np.sum, margins=True, fill_value=0)
  #  print("\ntable7=\n",table7.head(5))
    f.write("\n\n"+table7.to_string())

    table8 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['date','productgroup',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
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

    print("\n\nSales Prediction results written to spreadsheet",cfg.outxlsfile,"\n\n")
    f.write("\n\nSales Prediction results written to spreadsheet:"+str(cfg.outxlsfile)+"\n\n")
    
   
##############################################################################


    f.close()
    return

    

if __name__ == '__main__':
    main()
