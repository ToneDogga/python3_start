# Generic scikit-learn python3 regression
# Brand strength index model using regression
# written by Anthony Paech  14/12/19
#
# Idea:
# import from IRI weekly scan data for woolworths (or Coles) for a category like jams
# identify main competitors
# features:
# 1 try with all jam SKUs, then by individual SKU
# try with BB brand as predictor.  then try each brand
#
#  X=
# 1.St dalfour total weekly unit sales per store
# 2.st dalfour on promotion or not
# 3.Bon Maman total weekly units sales per store
# 4.BM on promotion or not
# 5.Cottees total weekly unit sales per store
# 6.Cottees total on Promotion or not
# 7.other brands in the category?
#
# y=
#   Beerenberg total weekly sales per store
#
# BB on promotion or not
# Xmas week?
#
# linear regression prediction:
# Beerenberg total weekly sales per store
#
# the r squared relationship of the predicted sales as opposed to actual sales on the testing set
# indicates the varibility of the model
#
# good brand strength by Beerenberg would indicate low or none drop in sales when competitors are on promotion
# check the correlation matrix between each competitors unit sales on and off promotion
#
#
# Two kinds.  there is classification of discrete classes and regression. This type is regression where the class ''y'' is a continuous variable
#
# Pseudocode
# load config file
# Data preparation and training sets generation
#   Load excel
#       Take a excel spreadsheet that is a slice of the salestrans.xlsx
#       filter by GLSET - eg "NAT" or "SHP"
#       and range of dates in excel prior
#       import into pandas
#
#   Feature building
#       import other supporting datasets from excel
#       index and join
#       collect and create all fields that are relevant
#       combine features over time or together to find accurate new sources
#       day_deltas instead of dates
#
#   Select columns as features
#
#   Vectorize non numeric features
#
#   Bin features if necessary
#       Cut the X dataframe of classes into a number of discrete bins    
#       use from collections import counter to check balance of distribution of the feature
#
#   Cleaning
#       clean or remove nans,infs and wrong type data
#       or Imput missing data
#
#   Strip out the target class column the dataframe Y from the features
#
#   Encode
#       Turn categorical features into numerical values if necessary
#       save the encoding
#
#   Scaling
#       to use SVC and SDG, the features need to be scaled to get an accurate result
#       Use sklearn.preprocessing StandardScaler
#       save the scaler
#
#   Smoothing
#       To avoid overfitting, smooth the features based on historial rhythms and seasonality of products, productgroups and customers
#
#   Document Feature creation
#
#   this final pandas dateframe is called X , the dataframe y is the target 1D dataframe of result for each row that turn into classes 
#
#   send pandas dataframe X to_numpy() array
#
#   send pandas dataframe y to_numpy() array
#
#   split into X_train, X_test, y_train, y_test from sklearn.model_selection train_test_split
# 
# Model training, evaluation and selection stage
#   Choose the right classifier algorithim
#       start with logistic regression SGD
#       Naive Bayes
#       Linear SVC
#       SVM (different kernels)
#       Random forest
#       Test each with different regulatisation using gridsearchCV
#
#   Setup grid_parameters
#       run using GridSearchCV
#
#   Evaluate
#       use mean squared error
#       use mean absolute error
#       use best_params_
#       use best_estimator_
#       use score
#       use predict
#       use predict_proba
#       use classification_report from sklearn.metrics
#       use confusion_matrix and AUC_ROC on binary results
#       
#   Feature ranking
#       Estimate the accuracy on each feature
#       using SVC, cross_val_score  (p224)
#       test each feature set for accuracy. reduce dimentionality as much as possible to avoid over fitting.

#   Reduce overfitting
#       Cross validation
#       Regulatization
#       simplify
#       balance specific data trained features with wider data features to avoid overfitting 
#
#   Diagnose overfitting and underfitting
#       use Scikit-learn learning_curve package and plot learning curve
#
#   Load and save and reuse models
#       save and load the encoding
#       save and load the Scaler
#       save and load the regressor
#
#   Get model testing performance > 0.90
#       Make predictions again y_test using the model and .predict(X_test,..)
#
#   monitor model performance
#       use r2_score from sklearn.metrics for regression ground truth
#       use correlation matrix and scatter matrix plots
#
###########################################################################################################3



#!/usr/bin/python3
from __future__ import print_function
from __future__ import division


import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix


import csv
import sys
import datetime as dt
import joblib
import pickle

import brand_strength_cfg as cfg

from dateutil.relativedelta import relativedelta


import matplotlib.pyplot as plt

import timeit

from collections import Counter,OrderedDict
    
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#import gc

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, MinMaxScaler, minmax_scale

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC

   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


from sklearn.pipeline import Pipeline

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning  

#from rfpimp import permutation_importances

#import eli5
#from eli5.sklearn import PermutationImportance

#from rfpimp import permutation_importances

###############################################################################################






def read_excel(filename,rows):
    xls = pd.read_excel(filename,sheet_name="Sheet1",header=1,index_col=None)    #'salestransslice1.xlsx')
    if rows==-1:
        return xls   #.parse(xls.sheet_names[0])
    else:        
        return xls.head(rows)





def date_deltas(df):
    df['date_encode'] = df['woolworths_scan_week'].map(dt.datetime.toordinal).astype(int)
    #    j["date"] = j.date_encode.map(dt.datetime.fromordinal)

 #   df['week_delta'] = (df.date.max()-df.date).dt.days.astype(int)/7#

    df['day_delta'] = (df.woolworths_scan_week-df.woolworths_scan_week.min()).dt.days.astype(int)
 #   df['week_delta'] = (df.date-df.date.min()).dt.days.astype(int)/7
##    df['two_week_delta'] = (df.date-df.date.min()).dt.days.astype(int)/14
#    df['month_delta'] = (df.date-df.date.min()).dt.days.astype(int)/30.416

 
 #   maxdate=df.day_delta.max().astype(int)
 #   df['day_of_year'] = df['date'].dt.dayofyear
 #   df['month_of_year'] = df['date'].dt.month
  #  df['day_of_month'] = df['date'].dt.dayofmonth
 
 #   df['week_of_year'] = df['date'].dt.weekofyear
 #   df['year']=df['date'].dt.year
    return df




def promotions(X_df):
    X_df["GMV"]=round((X_df["salesval"]-X_df["costval"])/(X_df["qty"]+0.0001),2)
    return X_df
       
def order_delta(df):
#    df2=df.sort_values(by=["code_encode","prod_encode","day_delta"],ascending=[True,True,False])   #,inplace=True)
    df2=df.sort_values(by=["code_encode","product","day_delta"],ascending=[True,True,False])   #,inplace=True)

    df2["day_order_delta"]=round(df2.day_delta.diff(periods=-1),0)

   # df["last_order_qty"]=0.01
    #df1=pd.DataFrame()
    df3=df2.copy(deep=True)
    
    cust_list=list(set(df2["code_encode"].tolist()))
 #   prod_list=list(set(df2["prod_encode"].tolist()))
    product_list=list(set(df2["product"].tolist()))

    cust_list_len=len(cust_list)
    product_list_len=len(product_list)

 #   scaled, active_bins,correction_factor=smoothing_distributions(df2,prod_list,cfg.bins)

    print("Read in excel product sales distributions for all customers-",cfg.scalerdump)
 #   f.write("Read in excel product sales distributions for all customers-"+cfg.scalerdump)

    scaler_df=pd.read_excel(cfg.scalerdump,header=0,index_col=None)
    print(scaler_df.columns)

 #   prod_list=list(set(scaler_df["prod_encode"].tolist()))
  #  product_list=list(set(scaler_df["product"].tolist()))
    product_list=list(set(df2["product"].tolist()))

 #   df.drop(['dd1'], axis=1,inplace=True)

    cust_list_len=len(cust_list)
    product_list_len=len(product_list)
#    df.drop(['product'], axis=1,inplace=True)


    
   #     scaled=scaler_df.to_numpy()
   # print('scaled=\n",scaled)
    
    i=0
    print("\n\n")
   # no_of_products=len(prod_list)
   # sales=Counter()
    for cust in cust_list:
        i+=1
        print("\rData Analysis Progress:{0:.1f}%".format(i/cust_list_len*100),end="\r",flush=True)
        for prod in product_list:
            last_order=((df2["product"]==prod) & (df2["code_encode"]==cust))    # & (df2.day_order_delta>=4) & (df2.day_order_delta<=45))

          #  df2["last_order_qty"]=df2[last_order].qty    #.shift(periods=-1,fill_value=0)
            df3.loc[df2[last_order].index,"last_order_upspd"]=df2[last_order].qty/df2[last_order].day_order_delta    #.shift(periods=-1,fill_value=0)

           # df1.loq=df2[last_order].qty.shift(periods=-1,fill_value=0)
          #  df1.loc[df2.index,"last_order_qty"]=df2[last_order].qty
        
       #    df1.loc[df.index,"last_order_upspd"]=df[last_order].qty/(df.day_order_delta+0.001)
          #  df2.drop(df2[(df2["last_order_qty"] == 0)].index,inplace=True)

    df3.drop(df3[(df3["day_order_delta"]<=0) | (df3["last_order_upspd"]<=0)].index,inplace=True)

#################################
    
    print("\n\nAdd bins on day_delta:",cfg.bins)
    #df3["bin_no"]=pd.qcut(df3["day_delta"],q=cfg.noofbins,labels=range(cfg.noofbins))
    df3["bin_no"]=pd.cut(df3["day_delta"],bins=cfg.bins,labels=range(len(cfg.bins)-1))





    
    k=0
    df3["prod_scaler"]=0.0


    product_list=list(set(scaler_df["product"].tolist()))
    product_list_len=len(product_list)
    scaler_df.drop(['product'], axis=1,inplace=True)
    scaler=scaler_df.to_numpy()
    print("\n\n")
    for prod in product_list:
        prod_mask=((df3["product"]==prod))
        binnumber_array=df3[prod_mask].bin_no
        j=0
        for abin in binnumber_array:
            df3.at[df3[prod_mask].index[j], 'prod_scaler']= scaler[k,abin]
            j+=1
        k+=1
        print("\rScaling Progress:{0:.1f}%".format(k/product_list_len*100),end="\r",flush=True)


    print("\n\n")
    psmean=df3["prod_scaler"].mean()

####################################################################3
    #  calculate scaling  scaled_updpd = last_order_upsdp +0.5+ prod_scaler * sensitivity_constant
    # prod_scaler is between 0 and 1.  we add the 0.5 to make the scaler 0.5 to 1.5 
#   df3["scaled_upspd"]=df3["last_order_upspd"]*df3["prod_scaler"]*cfg.rescale_constant

    df3["scaled_upspd"]=(df3["last_order_upspd"]*cfg.balance)+(((df3["prod_scaler"]-(psmean/2))*cfg.rescale_constant)*(1-cfg.balance))


    
 #   correction_factor=1/Xr_df["last_order_upspd"].mean()
    print("scaled upspd correction factor=",cfg.rescale_constant)
    print("balance factor between local data and general data=",cfg.balance)
    

    print("\nfinal df3 shape",df3.shape,"\n",df3.columns)
    
    return df3   #,correction_factor


def remove_non_trans(df):
    df["dd1"]=(df.qty<=0.0)  #  | df.order_delta==0.0)
    df.drop(df[df.dd1==True].index,inplace=True)
    df.drop(['dd1'], axis=1,inplace=True)
    return df



###########################################################################################33

def main():
   
    f=open(cfg.outfile,"w")

    print("\n\nBrand strength Index - By Anthony Paech 14/12/19\n")
#    print("Loading",sys.argv[1],"into pandas for processing.....")
    print("Loading",cfg.infilename,"into pandas for processing.....")

    print("Results Reported to",cfg.outfile)

    f.write("\n\nBrand Strength Index - By Anthony Paech 14/12/19\n\n")
 #   f.write("Loading "+sys.argv[1]+" into pandas for processing.....\n")
    f.write("Loading "+cfg.infilename+" into pandas for processing.....\n")

    f.write("Results Reported to "+cfg.outfile+"\n")

##    df=read_excel(sys.argv[1],-1)  # -1 means all rows
##    if df.empty:
##        print(sys.argv[1],"Not found.")
##        sys.exit()

    df=read_excel(cfg.infilename,cfg.importrows)  # -1 means all rows
    if df.empty:
        print(cfg.infilename,"Not found. Check brand_strength_cfg.py file")
        sys.exit()


   # print(df.head(10))
    print("data import shape:",df.shape)
    print("data import column names:",df.columns)
    f.write("data import shape:"+str(df.shape)+"\n")




    X_df=df.iloc[:,0:25]
      
    
    del df  # clear memory 

    print("Imported into pandas=\n",X_df.columns,"\n",X_df.shape)    #head(10))


    
##################################################33
##   # Remove extranous fields
##    b4=X_df.shape[1]
##    print("Remove columns not needed.")
###    X_df.drop(columns=["cat","code","costval","glset","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"],inplace=True)
##    X_df.drop(columns=cfg.excludecols,inplace=True)
##
##    print("Columns deleted:",b4-X_df.shape[1]) 
##
##    print("Delete rows for productgroup <10 or >15.")
##    b4=X_df.shape[0]
##    X_df.drop(X_df[(X_df["productgroup"]<10) | (X_df["productgroup"]>15)].index,inplace=True)
##    print("Rows deleted:",b4-X_df.shape[0])
##    
##
##    print("Prepare data....")
##    X_df.dropna(inplace=True)
##
##    X_df=date_deltas(X_df)
##
##    X_df=promotions(X_df)    # calculate the GMV (Gross margin value) as salesval-costval.  Used to highlight promotions
##    X_df.drop(columns=["salesval","costval"],inplace=True)
##    
####    label_encoder=LabelEncoder()
####    X_df["prod_encode"] = label_encoder.fit_transform(X_df["product"].to_numpy())
####    joblib.dump(label_encoder,open(cfg.product_encode_save,"wb"))
#### #   X_df.drop(columns=["product"],inplace=True)
##
##    label_encoder=LabelEncoder()
##    X_df["code_encode"] = label_encoder.fit_transform(X_df["code"].to_numpy())
##    joblib.dump(label_encoder,open(cfg.code_encode_save,"wb"))
##    X_df.drop(columns=["code"],inplace=True)
##    print(X_df.columns)
##
##
#####################################################
##    
##    #Xr_df = Xr_df[["prod_encode","qty","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]]  
###    Xr_df = Xr_df[["prod_encode","qty","productgroup","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]]  
##    Xc_df = X_df[cfg.featureorder_c]    # for classifier
##    Xr_df = X_df[cfg.featureorder_r]    # for regression
## #   print(X_df.columns)
##
##    del X_df   # clear memory
##    Xr_df=order_delta(Xr_df)
##  #  Xc_df=order_delta(Xc_df)
##
##    label_encoder=LabelEncoder()
##    Xr_df["prod_encode"] = label_encoder.fit_transform(Xr_df["product"].to_numpy())
##    joblib.dump(label_encoder,open(cfg.product_encode_save,"wb"))
## #   Xr_df.drop(columns=["product"],inplace=True)
##

    X_df["date"]=pd.to_datetime(X_df["woolworths_scan_week"])
    X_df['date_encode'] = X_df['date'].map(dt.datetime.toordinal).astype(int)
  #  X_df['day_delta'] = (X_df.date-X_df.date.min()).dt.days.astype(int)
    X_df['day_delta'] = (X_df.date.max()-X_df.date).dt.days.astype(int)



    X_df.drop(columns=["date","date_encode","woolworths_scan_week","bb_price","sd_price","c_price","bm_price","bb_upspw_incremental","sd_upspw_incremental","c_upspw_incremental","bm_upspw_incremental","bb_on_promo","sd_on_promo","bm_on_promo","c_on_promo"],inplace=True)
 
    #print(sales)
    X_df.dropna(inplace=True)
 #   Xc_df.dropna(inplace=True)

    print(X_df.columns)
    print("\nX_df=\n",X_df)


##    print("\nlast_order_upspd mean=",Xr_df["last_order_upspd"].mean(),"median=",Xr_df["last_order_upspd"].median(),"last_order_upspd stdev=",Xr_df["last_order_upspd"].std())
##    print("scaler mean=",Xr_df["prod_scaler"].mean(),"scaler median=",Xr_df["prod_scaler"].median(),"scaler stdev=",Xr_df["prod_scaler"].std())
##    print("scaled_upspd mean=",Xr_df["scaled_upspd"].mean(),"scaled_upspd median=",Xr_df["scaled_upspd"].median(),"scaled upspd stdev=",Xr_df["scaled_upspd"].std())
##    print("\n")
##    #print("scale sensitivity constant=",cfg.sensitivity_constant)
##
##  #  X = X[["code_encode","prod_encode","productgroup","week_of_year","order_delta"]]  
##    
##
##    Xr_df=remove_non_trans(Xr_df)
##  #  Xc_df=remove_non_trans(Xc_df)
##
##
##    Xr_df.to_csv(cfg.datasetworking,header=True,index=False)
##
##    Xr_df.drop(columns=["date"],inplace=True)
##    Xc_df.drop(columns=["date"],inplace=True)
##
## #   Xr_df=remove_poor_sellers(X_df,sales,cfg.minqty)
##    
##    Xr_df=cleanout_slow_sellers(Xr_df,cfg.mintransactions)   # any product with 16 or less transactions in the dataset is removed here
##  #  Xc_df=cleanout_slow_sellers(Xc_df,cfg.mintransactions)   # any product with 16 or less transactions in the dataset is removed here
##
##    Xr_df.drop(columns=["product"],inplace=True)
######################################################################

 #   important_attributes=["bb_total_upspw","sd_total_upspw","bm_total_upspw","c_total_upspw"]
  #  scatter_matrix(X_test[important_attributes],alpha=0.2,figsize=(12,9))
  #  scatter_matrix(X_df[important_attributes],alpha=0.2,figsize=(12,9))
  #  plt.show()


  #  important_attributes=["bb_promo_disc","sd_promo_disc","bm_promo_disc","c_promo_disc","bb_total_upspw","sd_total_upspw","bm_total_upspw","c_total_upspw"]
    important_attributes=["bb_promo_disc","sd_promo_disc","bm_promo_disc","bb_total_upspw","sd_total_upspw","bm_total_upspw"]

    scatter_matrix(X_df[important_attributes],alpha=0.2,figsize=(12,9))
    plt.show()


 #   important_attributes2=["beerenberg_upspw_incremental","st_dalfour_upspw_incremental","bon_maman_upspw_incremental","cottees_upspw_incremental"]

    
    corr_matrix=X_df.corr()   #.sort_values(ascending=False)   #[important_attributes])   #important_attributes2,method="pearson")
    print("\n\nCorrelations:\n",corr_matrix,"\n\n")


    sarraydf = pd.DataFrame (corr_matrix)

###### save to xlsx file
    print("Correlation matrix array saved to",cfg.scalerdump1)
    sarraydf.to_excel(cfg.scalerdump1, index=True)



############################################3


    y=X_df["bb_upspw_baseline"].to_numpy()  # regression
    y2=X_df["bb_total_upspw"].to_numpy()  # regression
 
    
    X_df.drop(columns=["bb_upspw_baseline","bb_total_upspw"],inplace=True)

  #  yc=Xc_df["prod_encode"].to_numpy()   # classification

 
#    X_df['day_delta'] = (X_df["woolworths_scan_week"].pd.to_datetime-X_df["woolworths_scan_week"].pd.to_datetime.min()).dt.days.astype(int)
   

  #  X_df.drop(columns=["last_weeks_product","last_two_weeks_product","last_month_product","qty"],inplace=True)
 #   X_df.drop(columns=["beerenberg_upspw_baseline","woolworths_scan_week"],inplace=True)
##
##  #  X_df=X_df[columns=[".astype(int)
##    counts=Counter(Xr_df.code_encode)   #.unique()
##    print("\nFrequency of customer codes:",counts)   #dict(zip(unique, counts)))
##    f.write("\nFrequency of customer codes:"+str(counts)+"\n")   #dict(zip(unique, counts)))
##
####    counts=Counter(Xr_df.product)   #.unique()
####    print("\nFrequency of products:",counts)   #dict(zip(unique, counts)))
####    f.write("\nFrequency of products:"+str(counts)+"\n")   #dict(zip(unique, counts)))
##
##    counts=Counter(Xr_df.prod_encode)   #.unique()
##    print("\nFrequency of product codes:",counts)   #dict(zip(unique, counts)))
##    f.write("\nFrequency of product codes:"+str(counts)+"\n")   #dict(zip(unique, counts)))
##
##    counts=Counter(Xr_df.productgroup)   #.unique()
##    print("\nFrequency of product groups:",counts)   #dict(zip(unique, counts)))
##    f.write("\nFrequency of product groups:"+str(counts)+"\n\n")   #dict(zip(unique, counts)))
##
##
##    Xr_df.drop(columns=["productgroup"],inplace=True)
##
##
##    fulldataset=Xr_df.copy(deep=True)
##
##    
 #   X_df.drop(columns=["date"],inplace=True)
  #  Xc_df.drop(columns=["prod_encode"],inplace=True)


##    Xr_df.to_csv(cfg.datasetpluspredict,header=True,index=False)


    print("X",X_df.columns)
    print("X_df cleaned. shape:",X_df.shape)
    f.write("X_df cleaned. shape:"+str(X_df.shape)+"\n")

    
    X=X_df.to_numpy()   # for regression on qty
##############################################################    

   # print("y=\n",y)

    print("X shape:",X.shape)
    f.write("X shape:"+str(X.shape)+"\n")
    print("y shape:",y.shape)
    f.write("r shape:"+str(y.shape)+"\n")

###########################################################

   

    print("Train/Test split and scaling.")
    f.write("Train/Test split and scaling.\n")


    X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)  # regression of qty
    X_train, X_test,y2_train,y2_test = train_test_split(X,y2, test_size=0.2,random_state=42)  # regression of qty

 #   Xc_train, Xc_test,yc_train,yc_test = train_test_split(Xc,yc, test_size=0.2,random_state=42)   # classification of prod_encode

    print("X_train.shape",X_train.shape)
    print("X_test.shape",X_test.shape)
    print("y_train.shape",y_train.shape)
    print("y_test.shape",y_test.shape)
    print("y2_train.shape",y2_train.shape)
    print("y2_test.shape",y2_test.shape)

    print("X",X_df.columns)

    #X2_test=np.copy(X_test)
    
    
##    dbd=pd.DataFrame(np.copy(X_test),columns=["sd_upspw_baseline","c_upspw_baseline","bm_upspw_baseline","bb_promo_disc","sd_promo_disc","c_promo_disc","bm_promo_disc","sd_total_upspw","c_total_upspw","bm_total_upspw","day_delta"])
##    dbd["bb_upspw_baseline"]=y_test
##    dbd["bb_total_upspw"]=y2_test
##
##    print("dbd=\n",dbd)
    
    ##    regressor = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, learning_rate='constant', eta0=0.01, max_iter=1000)
    ##    regressor.fit(X_train, y_train)
    ##
    ##    joblib.dump(regressor,open("SGDRegressorNS.p","wb"))
    ## 
    ##    predictions = regressor.predict(X_scaled_test)
    ##    #print(predictions)
    ##    print("SGDR regressor no scaling score:",regressor.score(X_scaled_test, y_test))





##        param_grid = {
##            "alpha": [1e-08,1e-07, 1e-06, 1e-05],
##            "penalty": [None, "l2"],
##            "eta0": [0.001, 0.005, 0.01,0.1],
##            "max_iter": [10000, 30000, 50000]
##        }
##
##        regressor = SGDRegressor(loss='squared_loss', learning_rate='constant')
##        grid_search = GridSearchCV(regressor, param_grid, cv=3, scoring='neg_mean_absolute_error',iid=False)
##        grid_search.fit(Xr_scaled_train, yr_train)
##
##
##
##    ##    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
##    ##    title = "Learning Curve (SVC)"
##    ##    # SVC is more expensive so we do a lower number of CV iterations:
##    ##    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
##    ##    estimator = SVC(gamma=0.001)
##    ##    plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),cv=cv, n_jobs=4)
##    ##
##    ##    plt.show()
##
##
##        joblib.dump(regressor,open(cfg.SGDR_save,"wb"))
##        print("SGDRegressor saved to:",cfg.SGDR_save)
##        f.write("SGDRegressor saved to:"+str(cfg.SGDR_save)+"\n")
##
##        print("SGDR best params",grid_search.best_params_)
##        f.write("SGDR best params:"+str(grid_search.best_params_)+"\n")
##
##        regressor_best = grid_search.best_estimator_
##       # regressor_best.score(X_test, y_test)
##
##        predictions = regressor_best.predict(Xr_scaled_test)
##
##        print("SGDR MSE=",mean_squared_error(yr_test, predictions))
##        print("SDGR MAE=",mean_absolute_error(yr_test, predictions))
##        print("SDGR R2=",r2_score(yr_test, predictions))
##        print("SDGR predictions=\n",predictions[:10])
##
##        f.write("SGDR MSE="+str(mean_squared_error(yr_test, predictions))+"\n")
##        f.write("SDGR MAE="+str(mean_absolute_error(yr_test, predictions))+"\n")
##        f.write("SDGR R2="+str(r2_score(y_test, predictions))+"\n")
##        f.write("SDGR predictions=\n"+str(predictions[:10].tolist())+"\n")
##
##
##
##        param_grid = {
##       #     "alpha": [1e-07, 1e-06, 1e-05],
##       #     "penalty": [None, "l2"],
##       #     "eta0": [0.001, 0.005, 0.01],
##       #     "max_iter": [3000, 10000, 30000]
##             "C":(0.1,1,10,100,1000,2000)
##        }
##        regressor = LinearSVR()
##        grid_search = GridSearchCV(regressor, param_grid, cv=5,scoring='neg_mean_absolute_error',iid=False)
##        grid_search.fit(Xr_scaled_train, yr_train)
##
##        joblib.dump(regressor,open(cfg.LSVR_save,"wb"))
##        print("LinearSVR saved to:",cfg.LSVR_save)
##        f.write("LinearSVR saved to:"+str(cfg.LSVR_save)+"\n")
##
##
##        print("Linear SVR best params",grid_search.best_params_)
##        f.write("Linear SVR best params"+str(grid_search.best_params_)+"\n")
##
##        regressor_best = grid_search.best_estimator_
##       # regressor_best.score(Xr_test, yr_test)
##
##        predictions = regressor_best.predict(Xr_scaled_test)
##
##        print("L SVR MSE=",mean_squared_error(yr_test, predictions))
##        print("L SVR MAE=",mean_absolute_error(yr_test, predictions))
##        print("L SVR R2=",r2_score(yr_test, predictions))
##        print("L SVR predictions=\n",predictions[:10])
##        
##        f.write("L SVR MSE="+str(mean_squared_error(yr_test, predictions))+"\n")
##        f.write("L SVR MAE="+str(mean_absolute_error(yr_test, predictions))+"\n")
##        f.write("L SVR R2="+str(r2_score(yr_test, predictions))+"\n")
##        f.write("L SVR predictions=\n"+str(predictions[:10].tolist())+"\n")
##
##
##        param_grid = {
##       #     "alpha": [1e-07, 1e-06, 1e-05],
##       #     "penalty": [None, "l2"],
##       #     "eta0": [0.001, 0.005, 0.01],
##       #     "max_iter": [3000, 10000, 30000]
##        #     "kernel":("linear"),
##             "epsilon":(0.01,0.02),
##             "C":(10,100,300)
##        }
##        regressor = SVR(kernel="linear")
##        grid_search = GridSearchCV(regressor, param_grid, cv=5,scoring='neg_mean_absolute_error')
##        grid_search.fit(Xr_scaled_train, yr_train)
##
##        joblib.dump(regressor,open(cfg.SVR_save,"wb"))
##        # regressor=joblib.load(open("SGDRegressor.p","rb"))
##        print("SVR saved to:",cfg.SVR_save)
##        f.write("SVR saved to:"+str(cfg.SVR_save)+"\n")
##
##
##        print("SVR best params",grid_search.best_params_)
##        f.write("SVR best params"+str(grid_search.best_params_)+"\n")
##
##        regressor_best = grid_search.best_estimator_
##       # regressor_best.score(X_test, y_test)
##
##        predictions = regressor_best.predict(Xr_scaled_test)
##
##        print("SVR MSE=",mean_squared_error(yr_test, predictions))
##        print("SVR MAE=",mean_absolute_error(yr_test, predictions))
##        print("SVR R2=",r2_score(yr_test, predictions))
##        print("SVR predictions=\n",predictions[:10])
##
##        f.write("SVR MSE="+str(mean_squared_error(yr_test, predictions))+"\n")
##        f.write("SVR MAE="+str(mean_absolute_error(yr_test, predictions))+"\n")
##        f.write("SVR R2="+str(r2_score(yr_test, predictions))+"\n")
##        f.write("SVR predictions=\n"+str(predictions[:10].tolist())+"\n")

    print("\nStarting RandomForestRegression on bb_baseline_upspw.")
    param_grid = {
   #     "alpha": [1e-07, 1e-06, 1e-05],
   #     "penalty": [None, "l2"],
   #     "eta0": [0.001, 0.005, 0.01],
   #     "max_iter": [3000, 10000, 30000]
    #     "kernel":("linear"),
         "max_depth":[20,30,40],
         "min_samples_split":[3,5]
    }
    regressor = RandomForestRegressor(n_estimators=cfg.RF_estimators,random_state=42,oob_score=True,bootstrap=True)
    grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error',verbose=True,n_jobs=-1)
    grid_search.fit(X_train, y_train)



    print("Random Forest best params",grid_search.best_params_)
    f.write("Random Forest best params"+str(grid_search.best_params_)+"\n")

    regressor_best = grid_search.best_estimator_

    joblib.dump(regressor_best,open(cfg.RFR_save,"wb"))
    print("RFR saved to:",cfg.RFR_save)
    f.write("RFR saved to:"+str(cfg.RFR_save)+"\n")

    
    print("RF best score:",regressor_best.score(X_test, y_test))
    f.write("RF best score:"+str(regressor_best.score(X_test, y_test))+"\n")

    predictions = regressor_best.predict(X_test)


    print("RF MSE=",mean_squared_error(y_test, predictions))
    print("RF MAE=",mean_absolute_error(y_test, predictions))
    print("RF OOB score=",regressor_best.oob_score_)
 #   print("RF OOB prediction=",regressor_best.oob_prediction_)
    print("RF R2=",r2_score(y_test, predictions))
#    print("RF predictions=\n",predictions[:10])
  #  print("RF predictions=\n",predictions)

    f.write("RF MSE="+str(mean_squared_error(y_test, predictions))+"\n")
    f.write("RF MAE="+str(mean_absolute_error(y_test, predictions))+"\n")
    f.write("RF OOB score="+str(regressor_best.oob_score_)+"\n")
 #   f.write("RF OOB prediction="+str(regressor_best.oob_prediction_)+"\n")
    f.write("RF R2="+str(r2_score(y_test, predictions))+"\n")
  #  f.write("RF predictions=\n"+str(predictions[:10].tolist())+"\n")
 #   f.write("RF predictions=\n"+str(predictions.tolist())+"\n")


    feature_sorted=np.argsort(regressor_best.feature_importances_)
    print("feature importance in order from weakest to strongest=",feature_sorted)
    f.write("feature importance in order from weakest to strongest="+str(feature_sorted.tolist())+"\n")

    cols=X_df[X_df.columns[feature_sorted]].columns
    print(cols)
    for name,score in zip(X_df.columns, regressor_best.feature_importances_):
        print("RF r2 score",name,score)
        f.write("RF r2 score "+str(name)+" = "+str(score)+"\n")

##################################################################################33333

    print("\nStarting RandomForestRegression on bb_total_upspw")
    param_grid = {
   #     "alpha": [1e-07, 1e-06, 1e-05],
   #     "penalty": [None, "l2"],
   #     "eta0": [0.001, 0.005, 0.01],
   #     "max_iter": [3000, 10000, 30000]
    #     "kernel":("linear"),
         "max_depth":[20,30,40],
         "min_samples_split":[3,5]
    }
    regressor = RandomForestRegressor(n_estimators=cfg.RF_estimators,random_state=42,oob_score=True,bootstrap=True)
    grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error',verbose=True,n_jobs=-1)
    grid_search.fit(X_train, y2_train)



    print("Random Forest best params",grid_search.best_params_)
    f.write("Random Forest best params"+str(grid_search.best_params_)+"\n")

    regressor_best = grid_search.best_estimator_

    joblib.dump(regressor_best,open(cfg.RFR_save,"wb"))
    print("RFR saved to:",cfg.RFR_save)
    f.write("RFR saved to:"+str(cfg.RFR_save)+"\n")

    
    print("RF best score:",regressor_best.score(X_test, y2_test))
    f.write("RF best score:"+str(regressor_best.score(X_test, y2_test))+"\n")

    predictions2 = regressor_best.predict(X_test)


    print("RF MSE=",mean_squared_error(y2_test, predictions))
    print("RF MAE=",mean_absolute_error(y2_test, predictions))
    print("RF OOB score=",regressor_best.oob_score_)
 #   print("RF OOB prediction=",regressor_best.oob_prediction_)
    print("RF R2=",r2_score(y2_test, predictions))
#    print("RF predictions=\n",predictions[:10])
  #  print("RF predictions=\n",predictions)

    f.write("RF MSE="+str(mean_squared_error(y2_test, predictions))+"\n")
    f.write("RF MAE="+str(mean_absolute_error(y2_test, predictions))+"\n")
    f.write("RF OOB score="+str(regressor_best.oob_score_)+"\n")
 #   f.write("RF OOB prediction="+str(regressor_best.oob_prediction_)+"\n")
    f.write("RF R2="+str(r2_score(y2_test, predictions))+"\n")
  #  f.write("RF predictions=\n"+str(predictions[:10].tolist())+"\n")
 #   f.write("RF predictions=\n"+str(predictions.tolist())+"\n")


    feature_sorted=np.argsort(regressor_best.feature_importances_)
    print("feature importance in order from weakest to strongest=",feature_sorted)
    f.write("feature importance in order from weakest to strongest="+str(feature_sorted.tolist())+"\n")

    cols=X_df[X_df.columns[feature_sorted]].columns
    print(cols)
    for name,score in zip(X_df.columns, regressor_best.feature_importances_):
        print("RF r2 score",name,score)
        f.write("RF r2 score "+str(name)+" = "+str(score)+"\n")
















#######################################################333
# visualisations

##    ccode_counts=Counter(X_df.code_encode)   #.unique()
##    print("\nFrequency of customer codes:",ccode_counts)   #dict(zip(unique, counts)))
##    f.write("\nFrequency of customer codes:"+str(ccode_counts)+"\n")   #dict(zip(unique, counts)))
##
####    counts=Counter(X_df.product)   #.unique()
####    print("\nFrequency of products:",counts)   #dict(zip(unique, counts)))
####    f.write("\nFrequency of products:"+str(counts)+"\n")   #dict(zip(unique, counts)))
##
##    pcode_counts=Counter(X_df.prod_encode)   #.unique()
##    print("\nFrequency of product codes:",pcode_counts)   #dict(zip(unique, counts)))
##    f.write("\nFrequency of product codes:"+str(pcode_counts)+"\n")   #dict(zip(unique, counts)))
##
##    pg_counts=Counter(X_df.productgroup)   #.unique()
##    print("\nFrequency of product groups:",pg_counts)   #dict(zip(unique, counts)))
##    f.write("\nFrequency of product groups:"+str(pg_counts)+"\n\n")   #dict(zip(unique, counts)))
##
##
##    print(Xr_df.columns)
##
##    print("\n\nCorrelations:\n",Xr_df.corr())
##    print("\n\n")
##
##
##
##
##    plt.hist(Xr_df["bin_no"].to_numpy(),histtype='stepfilled', density=False, bins=150) # density
##
###    sns.distplot(Xr_df["bin_no"],bins=150,rug=True)
##    plt.xlabel("Week Bin number")
##    plt.ylabel("Frequency")
##    plt.title("Number of transactions per week bin")
##  



# turns counter dictionary into a numpy array
 #   pc=pd.DataFrame(np.array(list(pcode_counts.items())),columns=["x","y"])
   # pc=np.array(list(pcode_counts.items()))[:,1]
 
##   # print("pc=",pc)
##   # #pdf=pd.DataFrame(pc).hist()
##    plt.bar(list(pcode_counts.keys()), list(pcode_counts.values()))
##  #  plt.hist(pc,bins=100)    #,histtype='stepfilled', density=False, bins=250) # density
##
##   # sns.distplot(pc["x"],kde=False, bins=200,rug=True)
##    plt.xlabel("Product Code")
##    plt.ylabel("Frequency")
##  #  plt.title("Product code frequency as a % of total transactions")
##    plt.show()
##
##    plt.bar(list(pg_counts.keys()), list(pg_counts.values()))
##    plt.xlabel("Product Groups")
##    plt.ylabel("Frequency")
##  #  plt.title("Product group frequency as a count of total transactions")
##    plt.show()
##
##    plt.bar(list(ccode_counts.keys()), list(ccode_counts.values()))
##
##    #sns.distplot(ccf["y"],kde=False,bins=200,rug=True)
##    plt.xlabel("Customer Code")
##    plt.ylabel("Frequency")
##  #  plt.title("Customer code frequency as a count of total transactions")
##    plt.show()
##
#############################################################################################3




    dbd=pd.DataFrame(np.copy(X_test),columns=["sd_upspw_baseline","c_upspw_baseline","bm_upspw_baseline","bb_promo_disc","sd_promo_disc","c_promo_disc","bm_promo_disc","sd_total_upspw","c_total_upspw","bm_total_upspw","day_delta"])
    dbd["bb_upspw_baseline"]=y_test
    dbd["bb_total_upspw"]=y2_test
    dbd["predict_bb_upspw_baseline"]=predictions.reshape(-1,1)
    dbd["predict_bb_total_upspw"]=predictions2.reshape(-1,1)

   # dbd2=pd.DataFrame(np.hstack((dbd.to_numpy(),predictions.reshape(-1,1))),columns=["day_order_delta","code","product","date","predict_qty"])



    print("dbd=\n",dbd)


   # dbd.sort_values(by=["date"],axis=0,ascending=[True],inplace=True)

    scatter_matrix(dbd[["bb_upspw_baseline","predict_bb_upspw_baseline","bb_total_upspw","predict_bb_total_upspw"]],alpha=0.2,figsize=(12,9))

 #   scatter_matrix(Xr_df["scaled_upspd","last_order_upspd","day_order_delta"],alpha=0.2,figsize=(12,9))
    plt.show()

    important_attributes=["beerenberg_total_upspw","st_dalfour_total_upspw","bon_maman_total_upspw","cottees_total_upspw"]
##
##    
    corr_matrix=dbd.corr()    #important_attributes)   #.sort_values(ascending=False)   #[important_attributes])   #important_attributes2,method="pearson")
##    print("\n\nCorrelations:\n",corr_matrix,"\n\n")
##  #  print(corr_matrix["st_dalfour_upspw_incremental"])
##
    print("\n\nCorrelations:\n",corr_matrix)
    print("\n\n")

    sarraydf = pd.DataFrame (corr_matrix)

###### save to xlsx file
    print("Correlation matrix array saved to",cfg.scalerdump2)
    sarraydf.to_excel(cfg.scalerdump2, index=True)
   
#
###################################################################

 #   scatter_matrix(Xr_df[["scaled_upspd","last_order_upspd","qty"]],alpha=0.2,figsize=(12,9))

 #   scatter_matrix(Xr_df["scaled_upspd","last_order_upspd","day_order_delta"],alpha=0.2,figsize=(12,9))
  #  plt.show()

   # print(predictions.shape)
  #  print("reshape")
    #print(predictions.reshape(-1,1).shape)


  #  X_test["beerenberg_predict_upspw_baseline"]=predictions   #pd.Series(predictions[:])    #.reshape(-1,1))

##
##    important_attributes=["st_dalfour_upspw_baseline","bon_maman_upspw_baseline","cottees_upspw_baseline"]
##  #  scatter_matrix(X_test[important_attributes],alpha=0.2,figsize=(12,9))
##    scatter_matrix(X_df[important_attributes],alpha=0.2,figsize=(12,9))
##
##    plt.show()
##
## #   important_attributes2=["beerenberg_upspw_incremental","st_dalfour_upspw_incremental","bon_maman_upspw_incremental","cottees_upspw_incremental"]
##
##    
##    corr_matrix=X_df.corr()   #.sort_values(ascending=False)   #[important_attributes])   #important_attributes2,method="pearson")
##    print("\n\nCorrelations:\n",corr_matrix,"\n\n")
##  #  print(corr_matrix["st_dalfour_upspw_incremental"])
##
##    print("\n\n")



##################################################################################

##    dbd=X_df.copy(deep=True)
##    regressor_best=joblib.load(open(cfg.RFR_save,"rb"))
##    print("RFR model loaded from:",cfg.RFR_save)
##    f.write("RFR model loaded from:"+str(cfg.RFR_save)+"\n")
##    predictions = regressor_best.predict(dbd)


############################################################3

    #j=pd.DataFrame(dbd,columns=["code_encode","prod_encode","date_encode","day_order_delta"])
##    encoder=joblib.load(open(cfg.code_encode_save,"rb"))
##    dbd["code"]=encoder.inverse_transform(dbd["code_encode"].astype(int).to_numpy())
##    dbd.drop(columns=["code_encode"],inplace=True)
##
##
##    encoder=joblib.load(open(cfg.product_encode_save,"rb"))
##    dbd["product"]=encoder.inverse_transform(dbd["prod_encode"].astype(int).to_numpy())
##    dbd.drop(columns=["prod_encode"],inplace=True)
##
##    dbd["date"] = dbd.date_encode.astype(int).map(dt.datetime.fromordinal)
##    dbd.drop(columns=["date_encode"],inplace=True)
##
##    dbd["qty"]=fulldataset["qty"]

   # dbd2=pd.DataFrame(np.hstack((dbd.to_numpy(),predictions.reshape(-1,1))),columns=["day_order_delta","code","product","date","predict_qty"])






  #  X_df.sort_values(by=["date_delta"],axis=0,ascending=[True],inplace=True)


  #  X_df.to_csv(cfg.datasetpluspredict,header=True,index=False)

 #   print("\n\nRaw data plus predictions saved to",cfg.datasetpluspredict,"...\n\n")



    f.close()
    return

    

if __name__ == '__main__':
    main()
