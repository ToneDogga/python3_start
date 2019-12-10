# Generic scikit-learn python3 regression
# written by Anthony Paech  4/12/19
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
#
###########################################################################################################3
#
#   Shop feature engineering
#   pyo_open?
#   temp?
#   rain forecast
#   School holidays
#   Public holidays
#   december?


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


import matplotlib.pyplot as plt

import timeit

from collections import Counter,OrderedDict
    
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import gc

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

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



def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))



def read_excel(filename,rows):
    xls = pd.ExcelFile(filename)    #'salestransslice1.xlsx')
    if rows==-1:
        return xls.parse(xls.sheet_names[0])
    else:        
        return xls.parse(xls.sheet_names[0]).head(rows)


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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,train_sizes=train_sizes)
            
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
   # fit_times_mean = np.mean(fit_times, axis=1)
   # fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
##    axes[1].grid()
##    axes[1].plot(train_sizes, fit_times_mean, 'o-')
##    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
##                         fit_times_mean + fit_times_std, alpha=0.1)
##    axes[1].set_xlabel("Training examples")
##    axes[1].set_ylabel("fit_times")
##    axes[1].set_title("Scalability of the model")
##
##    # Plot fit_time vs score
##    axes[2].grid()
##    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
##    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
##                         test_scores_mean + test_scores_std, alpha=0.1)
##    axes[2].set_xlabel("fit_times")
##    axes[2].set_ylabel("Score")
##    axes[2].set_title("Performance of the model")

    return plt




def date_deltas(df):
    df['date_encode'] = df['date'].map(dt.datetime.toordinal).astype(int)
    #    j["date"] = j.date_encode.map(dt.datetime.fromordinal)

 #   df['week_delta'] = (df.date.max()-df.date).dt.days.astype(int)/7#

    df['day_delta'] = (df.date-df.date.min()).dt.days.astype(int)
 #   df['week_delta'] = (df.date-df.date.min()).dt.days.astype(int)/7
##    df['two_week_delta'] = (df.date-df.date.min()).dt.days.astype(int)/14
#    df['month_delta'] = (df.date-df.date.min()).dt.days.astype(int)/30.416

 
 #   maxdate=df.day_delta.max().astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month_of_year'] = df['date'].dt.month
  #  df['day_of_month'] = df['date'].dt.dayofmonth
 
    df['week_of_year'] = df['date'].dt.weekofyear
    df['year']=df['date'].dt.year
    return df


def cleanout_slow_sellers(df,n):
    #n=4
    
  #  print(df.shape,"- Clean out low sellers less than or equal",n,"transactions in dataset")
    c=Counter(df["prod_encode"])
  #  print("n=",n,"=>,",c)
    lowtrans=list(c.most_common()[:-n-1:-1])
    h=[i[0] for i in lowtrans]
    print("prod_encodes to be removed from dataset=",h)
    j=0
    for i in h:
    #    mask=df["prod_encode"]==h[j]
        b4=df.shape[0]
        df.drop(df[(df["prod_encode"]==h[j])].index,inplace=True)
       # df.drop(df[(df(df["prod_encode"]==h[j])].index,inplace=True)
        #print("deleted",i,"->",b4-df.shape[0],"rows deleted...")
        j+=1
    print(df.shape)
    return df


def shift_reference(df,timeperiod,negshift,colname):
    number_of_rows=df.shape[0]
    df2=pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'])  
    df.sort_values(by=["date"],ascending=[False],inplace=True)

    newest_date=df.date.max()
    oldest_date=df.date.min()
    print("data from",oldest_date,"to",newest_date)

    if timeperiod=="days":
        print("days shifting",timeperiod,negshift,colname)
        previous=newest_date+ relativedelta(days=negshift)
    elif timeperiod=="weeks":
        print("weeks shifting",timeperiod,negshift,colname)
        previous=newest_date+ relativedelta(weeks=negshift)
    elif timeperiod=="months":
        print("months shifting",timeperiod,negshift,colname)
        previous=newest_date+ relativedelta(months=negshift)
    elif timeperiod=="years":
        print("years shifting",timeperiod,negshift,colname)
        previous=newest_date+ relativedelta(years=negshift)
    else:
        print("date error!. timeperiod=",timeperiod)
        previous=newest_date+ relativedelta(weeks=negshift)

    print("previous=",previous)    
    mask = (df['date'] >= oldest_date) & (df['date'] <= previous)
  #  print("newest=",newest_date)
  #  print("oldest=",oldest_date)
  #  print("pw=",previous_week)
    df2 = df.loc[mask]
  #  print("df2=\n",df2)   #.head(20))
  #  print(df2.shape)
    rows_missing=number_of_rows-df2.shape[0]
   # print("rows missing=",rows_missing)
    df3=df2.iloc[:,0:3]
  #  print("df3=\n",df3)   #.head(20))
  #  print(df3.shape)
 
#   df3.append(pd.Series(name=datetime.datetime(2018, 1, 1)),ignore_index=True)
    listofseries = [pd.Series(['NaN','NaN','NaN'])] * rows_missing
    #df4.reindex(df3.index.tolist() + list(range(20, 40)))
  #  print("list of series=",listofseries)
    df4=df3.append(listofseries, ignore_index=True, verify_integrity=False, sort=None)

 #   print("df4=",df4)   #.tail(10))
    print("df4 shape=",df4.shape)
    df[colname+"_prod_encode"]=df4["prod_encode"].to_numpy()
  #  df[colname+"_productgroup"]=df4["productgroup"].to_numpy()
    df[colname+"_qty"]=df4["qty"].to_numpy()
 #   df.append(pandas.Series(), ignore_index=True)
   # df3["product","qty"]
    return df


def smoothing_distributions(df,prod_list,noofbins):
    #  create a dictionary of prod_encodes as keys and the values are a counter is the day_delta as a key and the value is the sum of the qty sales for each day
    e=df.groupby(["prod_encode",pd.qcut(df.day_delta,q=noofbins,labels=range(noofbins))],as_index=[False,False])[['qty']].agg('sum').fillna(0)
    f=e.unstack(fill_value=0.0)
 #   print("f shape=",f.shape)
    sumf=f.qty.sum(axis=1).to_numpy()    #.reshape(-1,1)     #.tolist()
    #g=f.append(f,sumf.tolist())
   # print(sumf)
  #  print("sumf shape=",sumf.shape)
    i=0
    smoothing=[]   #np.empty((1,1),dtype=float)   #empty((len(prod_list),noofbins))    #np.zeros((len(prod_list,noofbins),dtype=float)
  #  print("smoothing shpae",smoothing.shape)
    for prod in prod_list:
        row=f.iloc[i,:].to_numpy()
        brkdwn=row/sumf[i]   #.reshape(-1,1)
     #   print("brk",brkdwn,"brkdwn shape=",brkdwn.shape)
        smoothing.append(brkdwn)
    #    print("smoothing shape=",smoothing.shape)
        i+=1
  #  print("smoothing=",smoothing)
  #  print("smoothing len=",len(smoothing))
    s=np.asarray(smoothing)
   # print(s[0:100])
    # normalise around zero.
    scale( s, axis=1, with_mean=False, with_std=True, copy=False )
    
   # print("scaled s",s[0:100])
   # print(s.shape,s.min(),s.max(),s.mean(),s.sum(axis=1))
    return s


       
def order_delta(df):
    df2=df.sort_values(by=["code_encode","prod_encode","day_delta"],ascending=[True,True,False])   #,inplace=True)

    df2["day_order_delta"]=round(df2.day_delta.diff(periods=-1),0)

   # df["last_order_qty"]=0.01
    #df1=pd.DataFrame()
    df3=df2.copy(deep=True)
    
    cust_list=list(set(df2["code_encode"].tolist()))
    prod_list=list(set(df2["prod_encode"].tolist()))
    cust_list_len=len(cust_list)-1
    prod_list_len=len(prod_list)-1

    scaler=smoothing_distributions(df2,prod_list,cfg.noofbins)
    i=0
    print("\n\n\n\n")
   # no_of_products=len(prod_list)
   # sales=Counter()
    for cust in cust_list:
        print("\rData Analysis Progress:{0:.1f}%".format(i/cust_list_len*100),end="\r",flush=True)
        i+=1
        for prod in prod_list:
            last_order=((df2["prod_encode"]==prod) & (df2["code_encode"]==cust))    # & (df2.day_order_delta>=4) & (df2.day_order_delta<=45))

          #  df2["last_order_qty"]=df2[last_order].qty    #.shift(periods=-1,fill_value=0)
            df3.loc[df2[last_order].index,"last_order_upspd"]=df2[last_order].qty/df2[last_order].day_order_delta    #.shift(periods=-1,fill_value=0)

           # df1.loq=df2[last_order].qty.shift(periods=-1,fill_value=0)
          #  df1.loc[df2.index,"last_order_qty"]=df2[last_order].qty
        
       #    df1.loc[df.index,"last_order_upspd"]=df[last_order].qty/(df.day_order_delta+0.001)
          #  df2.drop(df2[(df2["last_order_qty"] == 0)].index,inplace=True)

    df3.drop(df3[(df3["day_order_delta"]<=0) | (df3["last_order_upspd"]<=0)].index,inplace=True)

#################################
    
    print("Add bins on day_delta:",cfg.bins)
    df3["bin_no"]=pd.qcut(df3["day_delta"],q=cfg.noofbins,labels=range(cfg.noofbins))
#    print(df3)
    
###################################

 
    print("add scaling...")
    k=0
    df3["scaler"]=0.0
   # df4=df3.copy(deep=True)

 #   print("scaler=\n",scaler)
 #   print("len scaler",len(scaler))
  #  print("-0.363680344 = scaler[101,30]",scaler[101,30])
  #  input("?")
    for prod in prod_list:
        print("\rScaling Progress:{0:.1f}%".format(k/prod_list_len*100),end="\r",flush=True)
        prod_mask=((df3["prod_encode"]==prod))
        binnumber_array=df3[prod_mask].bin_no
     #   print("binnumber_array=",binnumber_array)
     #   print("len=",df3[prod_mask].shape,"\ndf3=\n",df3[prod_mask])
      #  df4=df3[prod_mask]
        j=0
        #df4.scaler=scaler[i]
      #  i=df3[prod_mask].index
       # print("df3 index=",j)

        for abin in binnumber_array:
          #  print("abin=",abin,"i=",i,"scaler=",scaler[i,abin])
       #     df3[prod_mask].scaler.iloc[j] = scaler[i,abin]   #.apply(copy_scaler)
        #    df3.at[j, 'scaler'] = scaler[i,abin]
        #    df.set_value('Row_index', 'Column_name', value)
            df3.at[df3[prod_mask].index[j], 'scaler']= scaler[k,abin]
            j+=1
        k+=1
      #  print("df5=\n",df4.head(100))

    
  #  print(df3)
    
    print("\n\n\n")


####################################################################3
    #  Multiply out scaling  scaled_updpd = last_order_upsdp * scaler * sensitivity_constant

    df3["scaled_upspd"]=df3["last_order_upspd"]*df3["scaler"]*cfg.sensitivity_constant

   # print("final df3-=\n",df3)
    

    
    return df3


##def copy_scaler(scaler,df3):
##    return =scaler[abin,i]   


def remove_poor_sellers(df,sales,n):   
    #print(df2.shape)

   # n=int(round(no_of_products/5,0))   # bottom 20%
 #   temp = sorted(list(sales.items()), key=lambda x: x[1])
 #   sales.clear()
 #   sales.update(temp)
    lowtrans=list(sales.most_common()[:-n-1:-1])
   # print("product encodes sorted by unit sales=",sales)   #,"lowtrand",lowtrans)
    h=[i[0] for i in lowtrans]
   # print("prod_encodes to be removed from dataset=",h)
    j=0
    for i in h:
        b4=df.shape[0]
        df.drop(df[(df["prod_encode"]==h[j])].index,inplace=True)
      #  print("deleted",i,"->",b4-df.shape[0],"rows deleted...")
        j+=1
    print(df.shape)

    return df


def remove_non_trans(df):
    df["dd1"]=(df.qty<=0.0)  #  | df.order_delta==0.0)
    df.drop(df[df.dd1==True].index,inplace=True)
 #   df["dd2"]=(df.day_order_delta==0.0)   # | df.week_order_delta.any().astype(int)==0 | df.two_week_order_delta.any().astype(int)==0 | df.month_order_delta.any().astype(int)==0)
 #   df.drop(df[df.dd2==True].index,inplace=True)
  #  df.drop(['dd1','dd2'], axis=1,inplace=True)
    df.drop(['dd1'], axis=1,inplace=True)

    return df



###########################################################################################33

def main():
##    if(len(sys.argv) < 2 ) :
##        print("Usage : python generic_sk_regression.py spreadsheet.xlsx")
##        sys.exit()

##    product_encode_save="product_encode_save.pkl"
## #   LY_product_encode_save="LY_product_encode_save.p"
###    glset_encode_save="glset_encode_save.p"
##
##    scaler_save="scaler.pkl"
##    SGDR_save="SGDRegressor.pkl"
##    LSVR_save="LinearSVRegressor.pkl"
##    SVR_save="SVRegressor.pkl"
##    RFR_save="RFRegressor.pkl"
    
    f=open(cfg.outfile,"w")

    print("\n\nData Regression ensemble - By Anthony Paech 5/12/19\n")
#    print("Loading",sys.argv[1],"into pandas for processing.....")
    print("Loading",cfg.infilename,"into pandas for processing.....")

    print("Results Reported to",cfg.outfile)

    f.write("\n\nData Regression ensemble - By Anthony Paech 5/12/19\n\n")
 #   f.write("Loading "+sys.argv[1]+" into pandas for processing.....\n")
    f.write("Loading "+cfg.infilename+" into pandas for processing.....\n")

    f.write("Results Reported to "+cfg.outfile+"\n")

##    df=read_excel(sys.argv[1],-1)  # -1 means all rows
##    if df.empty:
##        print(sys.argv[1],"Not found.")
##        sys.exit()

    df=read_excel(cfg.infilename,cfg.importrows)  # -1 means all rows
    if df.empty:
        print(cfg.infilename,"Not found. Check sales_regression_cfg.py file")
        sys.exit()


   # print(df.head(10))
    print("data import shape:",df.shape)
    f.write("data import shape:"+str(df.shape)+"\n")

  #  print("\nClean data 1")

    #X_test = label_encoder.transform(test)
  #  df.drop(columns=["code","product"],inplace=True)
  #  df = df[["code_encode","cat","prod_encode","productgroup","week_of_year","order_delta","qty_bins"]]


  #  df.drop(columns=["docentrynum","refer","territory"],inplace=True)
   # reorder columns here ->  df = df[["code_encode","cat","prod_encode","productgroup","week_of_year","order_delta","qty_bins"]]



    print("columns dropped. shape:",df.shape)
    f.write("columns dropped.  shape:"+str(df.shape)+"\n")


    X_df=df.iloc[:,0:17]
    
  
    
   # del df  # clear memory 

    print("Imported into pandas=\n",X_df.columns,"\n",X_df.shape)    #head(10))


    
##################################################33
   # Remove extranous fields
    b4=X_df.shape[1]
    print("Remove columns not needed.")
#    X_df.drop(columns=["cat","code","costval","glset","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"],inplace=True)
    X_df.drop(columns=cfg.excludecols,inplace=True)

    print("Columns deleted:",b4-X_df.shape[1]) 

    print("Delete rows for productgroup <10 or >15.")
    b4=X_df.shape[0]
    X_df.drop(X_df[(X_df["productgroup"]<10) | (X_df["productgroup"]>15)].index,inplace=True)
    print("Rows deleted:",b4-X_df.shape[0])
    

    print("Prepare data....")
    X_df.dropna(inplace=True)

    X_df=date_deltas(X_df)
    #  encode "code", "location", "product","date","glset"
    
    label_encoder=LabelEncoder()
    X_df["prod_encode"] = label_encoder.fit_transform(X_df["product"].to_numpy())
    joblib.dump(label_encoder,open(cfg.product_encode_save,"wb"))
    X_df.drop(columns=["product"],inplace=True)

    label_encoder=LabelEncoder()
    X_df["code_encode"] = label_encoder.fit_transform(X_df["code"].to_numpy())
    joblib.dump(label_encoder,open(cfg.code_encode_save,"wb"))
    X_df.drop(columns=["code"],inplace=True)
    print(X_df.columns)


###################################################
    
    #Xr_df = Xr_df[["prod_encode","qty","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]]  
#    Xr_df = Xr_df[["prod_encode","qty","productgroup","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]]  
    Xc_df = X_df[cfg.featureorder_c]    # for classifier
    Xr_df = X_df[cfg.featureorder_r]    # for regression
 #   print(X_df.columns)


 #   X_df=shift_reference(X_df,"days",-1,"yesterday")
  #  X_df=shift_reference(X_df,"weeks",-1,"last_weeks")
  #  X_df=shift_reference(X_df,"weeks",-2,"last_two_weeks")
   # X_df=shift_reference(X_df,"weeks",-3,"last_three_weeks")
#
  #  X_df=shift_reference(X_df,"weeks",-4,"last_four_weeks")
#    X_df=shift_reference(X_df,"months",-1,"last_month")
  #  X_df=shift_reference(X_df,"years",-1,"last_year")

 #   print(X_df.head(20).to_string())
 #   print(X_df.tail(20).to_string())

  #  input("?")
    Xr_df=order_delta(Xr_df)
  #  Xc_df=order_delta(Xc_df)

    #print(sales)
    Xr_df.dropna(inplace=True)
 #   Xc_df.dropna(inplace=True)
 
    print("last_order_upspd mean=",Xr_df["last_order_upspd"].mean(),"last_order_upspd stdev=",Xr_df["last_order_upspd"].std())
    print("scaler mean=",Xr_df["scaler"].mean(),"scaler stdev=",Xr_df["scaler"].std())
    print("scaled_upspd mean=",Xr_df["scaled_upspd"].mean(),"scaled upspd stdev=",Xr_df["scaled_upspd"].std())
    print("scale sensitivity constant=",cfg.sensitivity_constant)
  #  X = X[["code_encode","cat","prod_encode","productgroup","week_of_year","order_delta"]]  
    

    Xr_df=remove_non_trans(Xr_df)
  #  Xc_df=remove_non_trans(Xc_df)


    Xr_df.to_csv(cfg.datasetworking,header=True,index=False)

    
##    X_df.sort_values(by=["date_encode"],ascending=[True],inplace=True)
##    l=list(np.sort(np.unique(X_df.date_encode)))
##    print(l)   #ist(set(X_df.date.tolist())).sort()
##    X_df.sort_values(by=["date"],ascending=[True],inplace=True)
##    l=list(np.sort(np.unique(X_df.date)))
##    print(l)   #ist(set(X_df.date.tolist())).sort()

    Xr_df.drop(columns=["date"],inplace=True)
    Xc_df.drop(columns=["date"],inplace=True)

 #   Xr_df=remove_poor_sellers(X_df,sales,cfg.minqty)
    
    Xr_df=cleanout_slow_sellers(Xr_df,cfg.mintransactions)   # any product with 16 or less transactions in the dataset is removed here
    Xc_df=cleanout_slow_sellers(Xc_df,cfg.mintransactions)   # any product with 16 or less transactions in the dataset is removed here


    yr=Xr_df["qty"].to_numpy()  # regression
    yc=Xc_df["prod_encode"].to_numpy()   # classification

    

  #  X_df.drop(columns=["last_weeks_product","last_two_weeks_product","last_month_product","qty"],inplace=True)
    Xr_df.drop(columns=["day_delta"],inplace=True)

  #  X_df=X_df[columns=[".astype(int)
    counts=Counter(Xr_df.code_encode)   #.unique()
    print("\nFrequency of customer codes:",counts)   #dict(zip(unique, counts)))
    f.write("\nFrequency of customer codes:"+str(counts)+"\n")   #dict(zip(unique, counts)))

    counts=Counter(Xr_df.prod_encode)   #.unique()
    print("\nFrequency of product codes:",counts)   #dict(zip(unique, counts)))
    f.write("\nFrequency of product codes:"+str(counts)+"\n")   #dict(zip(unique, counts)))

    counts=Counter(Xr_df.productgroup)   #.unique()
    print("\nFrequency of product groups:",counts)   #dict(zip(unique, counts)))
    f.write("\nFrequency of product groups:"+str(counts)+"\n\n")   #dict(zip(unique, counts)))


    Xr_df.drop(columns=["productgroup"],inplace=True)




   # print("X_df=\n",X_df)
   
 #   X_df.drop(columns=["prod_encode"],inplace=True)
    
#    Xc=Xc_df.to_numpy()   # for classification on prod_encode
# remove prod_encode column
#    Xc=Xc[:,1:]
#    print("Xc=\n",Xc)
#    print("\nyc=\n",yc)

    fulldataset=Xr_df.copy(deep=True)

    
    Xr_df.drop(columns=["qty","last_order_upspd","scaler","bin_no"],inplace=True)
    Xc_df.drop(columns=["prod_encode"],inplace=True)


    Xr_df.to_csv(cfg.datasetpluspredict,header=True,index=False)


    print("Xr",Xr_df.columns)
    print("Xr_df cleaned. shape:",Xr_df.shape)
    f.write("Xr_df cleaned. shape:"+str(Xr_df.shape)+"\n")

    print("Xc",Xc_df.columns)
    print("Xc_df cleaned. shape:",Xc_df.shape)
    f.write("Xc_df cleaned. shape:"+str(Xc_df.shape)+"\n")

    
    Xr=Xr_df.to_numpy()   # for regression on qty
    Xc=Xc_df.to_numpy()   # for classification on prod_encode
    
  #  print(collections.Counter(X))
  #  unique, counts = np.unique(X[:,0], return_counts=True)
  #  print("Frequency of product codes:",dict(zip(unique, counts)))
    
  #  print(X)  
##############################################################    

   # print("y=\n",y)

    print("Xr shape:",Xr.shape)
    f.write("Xr shape:"+str(Xr.shape)+"\n")
    print("yr shape:",yr.shape)
    f.write("yr shape:"+str(yr.shape)+"\n")
    print("Xc shape:",Xc.shape)
    f.write("Xc shape:"+str(Xc.shape)+"\n")
  #  print("Xc=\n",Xc)
    print("yc shape:",yc.shape)
    f.write("yc shape:"+str(yc.shape)+"\n")
   # print("yc=\n",yc)

###########################################################

   

    print("Train/Test split and scaling.")
    f.write("Train/Test split and scaling.\n")


    Xr_train, Xr_test,yr_train,yr_test = train_test_split(Xr,yr, test_size=0.2,random_state=42)  # regression of qty
    Xc_train, Xc_test,yc_train,yc_test = train_test_split(Xc,yc, test_size=0.2,random_state=42)   # classification of prod_encode

    print("Xr_train.shape",Xr_train.shape)

#  this is not needed for random forest, only for SVC and SGDC
##    scaler=StandardScaler()
##    scaler.fit(Xr_train)
##
##    joblib.dump(scaler,open(cfg.scaler_save,"wb"))
##    print("Scaler saved to:",cfg.scaler_save)
##    f.write("Scaler saved to:"+str(cfg.scaler_save)+"\n")
##
##    
##    Xr_scaled_train=scaler.transform(Xr_train)
##    Xr_scaled_test=scaler.transform(Xr_test)


##    if True:
##        pass
##    else:



        
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

    print("\nStarting RandomForestRegression")
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
    grid_search.fit(Xr_train, yr_train)



    print("Random Forest best params",grid_search.best_params_)
    f.write("Random Forest best params"+str(grid_search.best_params_)+"\n")

    regressor_best = grid_search.best_estimator_

    joblib.dump(regressor_best,open(cfg.RFR_save,"wb"))
    print("RFR saved to:",cfg.RFR_save)
    f.write("RFR saved to:"+str(cfg.RFR_save)+"\n")

    
    print("RF best score:",regressor_best.score(Xr_test, yr_test))
    f.write("RF best score:"+str(regressor_best.score(Xr_test, yr_test))+"\n")

    predictions = regressor_best.predict(Xr_test)

    print("RF MSE=",mean_squared_error(yr_test, predictions))
    print("RF MAE=",mean_absolute_error(yr_test, predictions))
    print("RF OOB score=",regressor_best.oob_score_)
    print("RF OOB prediction=",regressor_best.oob_prediction_)
    print("RF R2=",r2_score(yr_test, predictions))
    print("RF predictions=\n",predictions[:10])

    f.write("RF MSE="+str(mean_squared_error(yr_test, predictions))+"\n")
    f.write("RF MAE="+str(mean_absolute_error(yr_test, predictions))+"\n")
    f.write("RF OOB score="+str(regressor_best.oob_score_)+"\n")
    f.write("RF OOB prediction="+str(regressor_best.oob_prediction_)+"\n")
    f.write("RF R2="+str(r2_score(yr_test, predictions))+"\n")
    f.write("RF predictions=\n"+str(predictions[:10].tolist())+"\n")


 #   print("collect 1=",gc.collect())    
 


#    del predictions, scaler,grid_search

 #   print("collect 2=",gc.collect())    
    
 #   X_trunc=X[-1330:,:]    # -330
 #   y_trunc=y[-1330:]

 #   print("collect 3=",gc.collect())    
 
    
 #   RF_class=RandomForestClassifier(n_estimators=1000,criterion='gini',n_jobs=-1)
 #   RF_class.fit(X_trunc,y_trunc)
    feature_sorted=np.argsort(regressor_best.feature_importances_)
    print("feature importance in order from weakest to strongest=",feature_sorted)
    f.write("feature importance in order from weakest to strongest="+str(feature_sorted.tolist())+"\n")

   # for n in iter(feature_sorted):
   #     print(df.columns[n]])
    cols=Xr_df[Xr_df.columns[feature_sorted]].columns
    print(cols)
    for name,score in zip(Xr_df.columns, regressor_best.feature_importances_):
        print("RF r2 score",name,score)
        f.write("RF r2 score "+str(name)+" = "+str(score)+"\n")

 #   for name,score in zip(df.columns, feature_sorted):
 #       print("sorted r2 score",name,score)


  #  perm = PermutationImportance(rf, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
  #  perm_imp_eli5 = imp_df(cols, perm.feature_importances_)
 #   perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)
 #   print("Eli5 PI=",perm_imp_eli5)
 #   print("rfpimp PI=",perm_imp_rfpimp)
        
    # Select different number of top features
##    K = [3, 4, 5, 6,7]
##    for k in K:
##        top_K_features = feature_sorted[-k:]
##        X_k_selected = X_trunc[:, top_K_features]
##        # Estimate accuracy on the data set with k selected features
##        classifier = RandomForestClassifier(n_estimators=1000,random_state=42)
##        score_k_features = cross_val_score(classifier, X_k_selected, y_trunc).mean()
##        print('Score with the data set of top {0} features: {1:.2f}'.format(k, score_k_features))

##    print("\nStarting SGD Classification")
##
##    param_grid = {
##        "alpha": [1e-07, 1e-06, 1e-05],
##        "penalty": ["l2",None],
##        "eta0": [0.001, 0.005, 0.01],
##        "max_iter": [300,3000]
##    #     "kernel":("linear"),
##   #      "max_depth":[3,7,10,30,None],
##   #      "min_samples_split":[20,30]
##    }
##    classifier = SGDClassifier(loss="log", learning_rate="constant", eta0=0.01, fit_intercept=True, max_iter=300)    #n_estimators=cfg.SGDC_estimators,criterion="gini",random_state=42,n_jobs=-1)
##    grid_search = GridSearchCV(classifier, param_grid, cv=5,verbose=True,n_jobs=-1)
##    grid_search.fit(Xc_train, yc_train)
##
##    joblib.dump(classifier,open(cfg.SGDC_save,"wb"))
##    print("SGDC saved to:",cfg.SGDC_save)
##    f.write("SGDC saved to:"+str(cfg.SGDC_save)+"\n")
##
##
##    print("SGDClassifier best params",grid_search.best_params_)
##    f.write("SGDClassifier best params"+str(grid_search.best_params_)+"\n")
##
##    classifier_best = grid_search.best_estimator_
##    print("SGDC best score:",classifier_best.score(Xc_test, yc_test))
##    f.write("SGDC best score:"+str(classifier_best.score(Xc_test, yc_test))+"\n")

##    pos_prob=classifier_best.predict_proba(Xc_test)[:,1]
##
##    print("The ROC AUC on testing set is: {0:3f}".format(roc_auc_score(yc_test,pos_prob)))
##
##    predictions = classifier_best.predict(Xc_test)
##
####    print("RF MSE=",mean_squared_error(yc_test, predictions))
####    print("RF MAE=",mean_absolute_error(yc_test, predictions))
####    print("RF OOB score=",regressor_best.oob_score_)
####    print("RF OOB prediction=",regressor_best.oob_prediction_)
##    print("RFC R2=",r2_score(yc_test, predictions))
##    print("RFC predictions=\n",predictions[:10])
##
####    f.write("RF MSE="+str(mean_squared_error(yr_test, predictions))+"\n")
####    f.write("RF MAE="+str(mean_absolute_error(yr_test, predictions))+"\n")
####    f.write("RF OOB score="+str(regressor_best.oob_score_)+"\n")
####    f.write("RF OOB prediction="+str(regressor_best.oob_prediction_)+"\n")
##    f.write("RFC R2="+str(r2_score(yc_test, predictions))+"\n")
##    f.write("RFC predictions=\n"+str(predictions[:10].tolist())+"\n")


#########################################################################################3

##    print("\nStarting Random Forest Classification")
##
##    param_grid = {
##    #    "alpha": [1e-07, 1e-06, 1e-05],
##    #    "penalty": ["l2",None],
##    #    "eta0": [0.001, 0.005, 0.01],
##    #    "max_iter": [300,3000]
##    #     "kernel":("linear"),
##         "max_depth":[3,7,10,None],
##         "min_samples_split":[20,30,40]
##    }
##    classifier = RandomForestClassifier(n_estimators=cfg.RFC_estimators,criterion="gini",random_state=42,n_jobs=-1)
##    grid_search = GridSearchCV(classifier, param_grid, cv=5,verbose=True,n_jobs=-1)
##    grid_search.fit(Xc_train, yc_train)
##
##
##
##    print("RFClassifier best params",grid_search.best_params_)
##    f.write("RFClassifier best params"+str(grid_search.best_params_)+"\n")
##
##    classifier_best = grid_search.best_estimator_
##
##    joblib.dump(classifier_best,open(cfg.RFC_save,"wb"))
##    print("RFC saved to:",cfg.RFC_save)
##    f.write("RFC saved to:"+str(cfg.RFC_save)+"\n")
##
##    
##    print("RFC best score:",classifier_best.score(Xc_test, yc_test))
##    f.write("RFC best score:"+str(classifier_best.score(Xc_test, yc_test))+"\n")
##
##    pos_prob=classifier_best.predict_proba(Xc_test)[:,1]
##   # print("pos prob=",pos_prob)

###########################################################################33
# create predictions 
  #  one_year_ago = (dt.datetime.now()+relativedelta(years=-1)).strftime('%Y/%m/%d')
  #  print("one year ago",one_year_ago)

    dbd=Xr_df.copy(deep=True)
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

    dbd["qty"]=fulldataset["qty"]

    dbd["predict_qty"]=predictions.reshape(-1,1)
   # dbd2=pd.DataFrame(np.hstack((dbd.to_numpy(),predictions.reshape(-1,1))),columns=["day_order_delta","code","product","date","predict_qty"])

    dbd.sort_values(by=["date"],axis=0,ascending=[True],inplace=True)


    dbd.to_csv(cfg.datasetpluspredict,header=True,index=False)

    print("\n\nRaw data plus predictions saved to",cfg.datasetpluspredict,"...\n\n")

##################################################################################################    
##
##    last_year=(dbd["date"]>=one_year_ago)    # & (df2.day_order_delta>=4) & (df2.day_order_delta<=45))
##
##    dbd2=dbd.loc[last_year,:]
##
##    #dbd2=dbd[dbd["date">=one_year_ago]].index
##    print(dbd2)
##    print(dbd2.columns)
##
##  #  print("j=\n",j)


#####################################################################################

##
##  #  print("X_test=\n",X_test[:10,:],"y_predict:",predictions[:10].reshape(-1,1))
##    Xr_test=Xr_test[:,:4]
##
##   # print("Xr_test=\n",Xr_test)
##    
##    j=pd.DataFrame(Xr_test,columns=["code_encode","prod_encode","date_encode","day_order_delta"])
##  #  print(j)
##
##    encoder=joblib.load(open(cfg.code_encode_save,"rb"))
##    j["code"]=encoder.inverse_transform(j["code_encode"].astype(int).to_numpy())
##    j.drop(columns=["code_encode"],inplace=True)
##
##
##    encoder=joblib.load(open(cfg.product_encode_save,"rb"))
##    j["product"]=encoder.inverse_transform(j["prod_encode"].astype(int).to_numpy())
##    j.drop(columns=["prod_encode"],inplace=True)
##
##    j["date"] = j.date_encode.astype(int).map(dt.datetime.fromordinal)
##    j.drop(columns=["date_encode"],inplace=True)
##
##  #  print("j=\n",j)
##    
##
##
##    k=pd.DataFrame(np.hstack((j,predictions.reshape(-1,1))),columns=["day_order_delta","code","product","date","predict_qty"])
##
##  
## #   j["date"] = pd.Timestamp(dt.datetime.today()-j.day_delta.min()+j.day_delta)
##
###  dt.datetime.today().strftime("%d/%m/%Y")
## #   LY_product_encode_save=joblib.load(open("LY_product_encode_save.p","rb"))
## #   encoder2=joblib.load(open(glset_encode_save,"rb"))
## #   j["glset"]=encoder2.inverse_transform(j["glset_encode"].astype(int).to_numpy())
##
## #   print("k=\n",k[0:15].to_string())



##    
##    now = dt.datetime.now()   #.strftime('%d/%m/%Y')
## #   now_dt = datetime.strptime(now, '%Y/%m/%d').date()
##    print("now=",now)
##    dbd2["new_date"]=now  #.dt.date
##    dbd2['new_date'] = dbd2['new_date'] + pd.to_timedelta(dbd2['day_order_delta'], unit='d')
##    dbd2['predict_date']=dbd2['new_date'].dt.strftime('%Y/%m/%d')
##   # new_datetime_obj = datetime.strptime(orig_datetime_obj.strftime('%d-%m-%y'), '%d-%m-%y').date()
## 
##    dbd2= dbd2[["code","product","predict_date","predict_qty"]]
##    
##    dbd2.sort_values(by=["predict_date"],axis=0,ascending=[False],inplace=True)
##    dbd2["predict_qty"]=dbd2["predict_qty"].astype(float).round(0)    #{"predict_qty" : 1})
##    dbd2["predict_qty_ctnsof8"]=(dbd2["predict_qty"]/8).astype(float).round(0)
##    #   print(k)
##    print("Sales Qty Predictions=\n",dbd2[0:100].to_string())
##    
##   
###############################################################################3
##    #  create a pivot table of code, product, day delta and predicted qty and export back to excel
##
##    table = pd.pivot_table(dbd2, values='predict_qty', index=['product', 'predict_date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable=\n",table)
##    f.write("\n\n"+table.to_string())
##
##    table2 = pd.pivot_table(dbd2, values='predict_qty', index=['code', 'predict_date'],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable2=\n",table2)
##    f.write("\n\n"+table2.to_string())
##
##    table3 = pd.pivot_table(dbd2, values='predict_qty', index=['predict_date',"code"],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable3=\n",table3)
##    f.write("\n\n"+table3.to_string())
##
##    table4 = pd.pivot_table(dbd2, values='predict_qty', index=['predict_date',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable4=\n",table4)
##    f.write("\n\n"+table4.to_string())
##
##    table5 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['product', 'predict_date'],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable5=\n",table5)
##    f.write("\n\n"+table5.to_string())
##
##    table6 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['code', 'predict_date'],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable6=\n",table6)
##    f.write("\n\n"+table6.to_string())
##
##    table7 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['predict_date',"code"],columns=['product'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable7=\n",table7)
##    f.write("\n\n"+table7.to_string())
##
##    table8 = pd.pivot_table(dbd2, values='predict_qty_ctnsof8', index=['predict_date',"product"],columns=['code'], aggfunc=np.sum, margins=True, fill_value=0)
##    print("\ntable8=\n",table8)
##    f.write("\n\n"+table8.to_string())
##
##
##    with pd.ExcelWriter(cfg.outxlsfile) as writer:  # mode="a" for append
##        table.to_excel(writer,sheet_name="Sheet1")
##        table2.to_excel(writer,sheet_name="Sheet2")
##        table3.to_excel(writer,sheet_name="Sheet3")
##        table4.to_excel(writer,sheet_name="Sheet4")
##        table5.to_excel(writer,sheet_name="Sheet5")
##        table6.to_excel(writer,sheet_name="Sheet6")
##        table7.to_excel(writer,sheet_name="Sheet7")
##        table8.to_excel(writer,sheet_name="Sheet8")
##
##   
################################################################################


    f.close()
    return

    

if __name__ == '__main__':
    main()
