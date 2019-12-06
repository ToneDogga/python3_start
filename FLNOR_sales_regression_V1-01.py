# Generic scikit-learn python3 regression
# written by Anthony Paech  4/12/19
#
# Two kinds.  there is classification of discrete classes and regression. This type is regression where the class ''y'' is a continuous variable
#
# Pseudocode
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

from dateutil.relativedelta import relativedelta


import matplotlib.pyplot as plt

import timeit

from collections import Counter
    
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import gc

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC

   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

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

 #   df['week_delta'] = (df.date.max()-df.date).dt.days.astype(int)/7

    df['day_delta'] = (df.date-df.date.min()).dt.days.astype(int)
##    df['week_delta'] = (df.date-df.date.min()).dt.days.astype(int)/7
##    df['two_week_delta'] = (df.date-df.date.min()).dt.days.astype(int)/14
##    df['month_delta'] = (df.date-df.date.min()).dt.days.astype(int)/30.416

 
 #   maxdate=df.day_delta.max().astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month_of_year'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.weekofyear
    df['year']=df['date'].dt.year
    return df

def shift_last_years(df):

    df2=pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'])  
    df.sort_values(by=["date"],ascending=[False],inplace=True)

    newest_date=df.date.max()
    oldest_date=df.date.min()
    previous_week=newest_date+ relativedelta(weeks=-1)
    mask = (df['date'] >= oldest_date) & (df['date'] <= previous_week)
    print("newest=",newest_date)
    print("oldest=",oldest_date)
    print("pw=",previous_week)
    df = df.loc[mask]
    print(df.head(20))
    print(df.shape)
    df2=df.iloc[:,0:3]
    print("df2=",df2.head(10))
    df["last_weeks_product"]=df2["product"].to_numpy()
    df["last_weeks_qty"]=df2["qty"].to_numpy()
   # df2["product","qty"]
    return df

def order_delta(df):
    df.sort_values(by=["productgroup","product","day_delta"],ascending=[True,True,False],inplace=True)

    df["day_order_delta"]=round(df.day_delta.diff(periods=-1),0)
    df["previous_qty"]=df["qty"].shift(periods=-1,fill_value=0.0)
  #  df["weekly_previous_qty"]=df["qty"].shift(periods=-1,fill_value=0.0)/(df["day_order_delta"]*7)
  #  df["two_weekly_previous_qty"]=df["qty"].shift(periods=-1,fill_value=0.0)/(df["day_order_delta"]*14)
  #  df["monthly_previous_qty"]=df["qty"].shift(periods=-1,fill_value=0.0)/(df["day_order_delta"]*30)

   # df["test_qty"]=df["qty"]

  #  df["last_month_order_qty"]=df.query("day_order_delta<=30.0 & day_order_delta>14").qty.shift(periods=-1,fill_value=11.11)
  #  df["last_two_week_order_qty"]=df.query("day_order_delta<=14.0 & day_order_delta>7").qty.shift(periods=-1,fill_value=11.11)
  #  df["last_week_order_qty"]=df.query("day_order_delta<=7.0").qty.shift(periods=-1,fill_value=11.11)

    df1=pd.DataFrame()
   
    df["last_week_order_qty"]=0.0
    df["last_two_week_order_qty"]=0.0
    df["last_month_order_qty"]=0.0
    
    df2=df.copy(deep=True)
    
    prod_list=list(set(df["product"].tolist()))
    print("prod_list=",prod_list)
    for prod in prod_list:
        last_month_order=((df2["product"]==prod) & (df2.day_order_delta>21) & (df2.day_order_delta<=51))
      #  print(prod,"1=",df[last_month_order])
      #  df["last_month_order_qty"]=df[last_month_order].qty.shift(periods=-1,fill_value=11.11)
     #   print("lmo=\n",df[last_month_order].qty.shift(periods=-1,fill_value=11.11))
        df1.lmo=df2[last_month_order].qty.shift(periods=-1,fill_value=0)/4   #.to_numpy()
      #  print("lmo=\n",df1.lmo)
      #  df2 = df.assign(last_month_order_qty=df1.lmo)
        df2.loc[df1.lmo.index,"last_month_order_qty"]=df1.lmo
      #  df["last_month_order"]=lmo
      #  print(prod,"2=",df[last_month_order])

    for prod in prod_list:
        last_two_week_order=((df2["product"]==prod) & (df2.day_order_delta>7) & (df2.day_order_delta<=21))
      #  print(prod,"1=",df[last_two_week_order])
      #  df["last_month_order_qty"]=df[last_month_order].qty.shift(periods=-1,fill_value=11.11)
     #   print("lmo=\n",df[last_month_order].qty.shift(periods=-1,fill_value=11.11))
        df1.lmo=df2[last_two_week_order].qty.shift(periods=-1,fill_value=0)/2   #.to_numpy()
      #  print("lmo=\n",df1.lmo)
      #  df2 = df.assign(last_two_week_order_qty=df1.lmo)
      #  df["last_month_order"]=lmo
      #  print(prod,"2=",df[last_two_week_order])
        df2.loc[df1.lmo.index,"last_two_week_order_qty"]=df1.lmo

        
    for prod in prod_list:
        last_week_order=((df2["product"]==prod) & (df2.day_order_delta>0) & (df2.day_order_delta<=7))
       # print(prod,"1=",df[last_week_order])
      #  df["last_month_order_qty"]=df[last_month_order].qty.shift(periods=-1,fill_value=11.11)
     #   print("lmo=\n",df[last_month_order].qty.shift(periods=-1,fill_value=11.11))
        df1.lmo=df2[last_week_order].qty.shift(periods=-1,fill_value=0)   #.to_numpy()
       # print("lmo=\n",df1.lmo)
    #    df2 = df.assign(last_week_order_qty=df1.lmo)
      #  df["last_month_order"]=lmo
      #  print(prod,"2=",df)   #df[last_week_order])
        df2.loc[df1.lmo.index,"last_week_order_qty"]=df1.lmo

    print("last_week_order=\n",df2[(df2.day_order_delta>0) & (df2.day_order_delta<=7)])
    print("last_two_week_order=\n",df2[(df2.day_order_delta>7) & (df2.day_order_delta<=21)])
    print("last_month_order=\n",df2[(df2.day_order_delta>21) & (df2.day_order_delta<=51)])





   # df["last_two_week_order_qty"]=df.query("day_order_delta<=14.0 & day_order_delta>7").qty.shift(periods=-1,fill_value=11.11)
   # df["last_week_order_qty"]=df.query("day_order_delta<=7.0").qty.shift(periods=-1,fill_value=11.11)

    df2.day_order_delta.clip(lower=0,inplace=True)

  #  df.sort_values(by=["productgroup","product","day_delta"],ascending=[True,True,False],inplace=True)

 
##    df.sort_values(by=["productgroup","product","week_delta"],ascending=[True,True,False],inplace=True)
##    df["week_order_delta"]=round(df.week_delta.diff(periods=-1),0)
 #   df2.week_order_delta.clip(lower=0,inplace=True)
##
##    df.sort_values(by=["productgroup","product","two_week_delta"],ascending=[True,True,False],inplace=True)
##    df["two_week_order_delta"]=df.two_week_delta.diff(periods=-1)
 #   df2.two_week_order_delta.clip(lower=0,inplace=True)
## 
##    df.sort_values(by=["productgroup","product","month_delta"],ascending=[True,True,False],inplace=True)
##    df["month_order_delta"]=df.month_delta.diff(periods=-1)
 #   df2.month_order_delta.clip(lower=0,inplace=True)



  #  print(df2)
    return df2


def remove_non_trans(df):
    df["dd1"]=(df.qty<=0.0)  #  | df.order_delta==0.0)
    df.drop(df[df.dd1==True].index,inplace=True)
    df["dd2"]=(df.day_order_delta==0.0)   # | df.week_order_delta.any().astype(int)==0 | df.two_week_order_delta.any().astype(int)==0 | df.month_order_delta.any().astype(int)==0)
    df.drop(df[df.dd2==True].index,inplace=True)
    df.drop(['dd1','dd2'], axis=1,inplace=True)
    return df



###########################################################################################33

def main():
    if(len(sys.argv) < 2 ) :
        print("Usage : python generic_sk_regression.py spreadsheet.xlsx")
        sys.exit()

    product_encode_save="product_encode_save.pkl"
 #   LY_product_encode_save="LY_product_encode_save.p"
#    glset_encode_save="glset_encode_save.p"

    scaler_save="scaler.pkl"
    SGDR_save="SGDRegressor.pkl"
    LSVR_save="LinearSVRegressor.pkl"
    SVR_save="SVRegressor.pkl"
    RFR_save="RFRegressor.pkl"
    outfile="RegressionResults.txt"
    f=open(outfile,"w")

    print("\n\nData Regression ensemble - By Anthony Paech 5/12/19\n")
    print("Loading",sys.argv[1],"into pandas for processing.....")
    print("Results Reported to",outfile)

    f.write("\n\nData Regression ensemble - By Anthony Paech 5/12/19\n\n")
    f.write("Loading "+sys.argv[1]+" into pandas for processing.....\n")
    f.write("Results Reported to "+outfile+"\n")

    df=read_excel(sys.argv[1],-1)  # -1 means all rows
    if df.empty:
        print(sys.argv[1],"Not found.")
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

    print("Imported into pandas=\n",X_df.head(10))


    
##################################################33
   # Remove extranous fields
    X_df.drop(columns=["cat","code","costval","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"],inplace=True)


    #  encode "code", "location", "product","date","glset"
    

    #X_df["prod_encode"] = one_hot_encoder.fit_transform(X_df["product"].to_numpy())
  #  X_df["LY_prod_encode"] = label_encoder.fit_transform(X_df["LY_product"].to_numpy())
  #  joblib.dump(label_encoder,open(LY_product_encode_save,"wb"))


  #  X_df["code_encode"] = label_encoder.fit_transform(X_df["code"].to_numpy())
 #   X_df["location_encode"] = label_encoder.fit_transform(X_df["location"].to_numpy())
 #   X_df["glset_encode"] = label_encoder.fit_transform(X_df["glset"].to_numpy())
 #   joblib.dump(label_encoder,open(glset_encode_save,"wb"))


    X_df=date_deltas(X_df)
    X_df=shift_last_years(X_df)
    print(X_df)
    input("?")
    X_df=order_delta(X_df)
    X_df.dropna(inplace=True)
  #  X = X[["code_encode","cat","prod_encode","productgroup","week_of_year","order_delta"]]  
    

    X_df=remove_non_trans(X_df)


    X_df.drop(columns=["glset","date"],inplace=True)

    label_encoder=LabelEncoder()
    #one_hot_encoder=OneHotEncoder()
    X_df["prod_encode"] = label_encoder.fit_transform(X_df["product"].to_numpy())
    joblib.dump(label_encoder,open(product_encode_save,"wb"))

 #   X_df.drop(columns=["product"],inplace=True)


    y=X_df["qty"].to_numpy()
    # drop the y target out of X
  #  X_df["previous_qty"]=X_df["qty"]
    X_df.drop(columns=["product","qty"],inplace=True)




    print("X_df cleaned. shape:",X_df.shape)
    f.write("X_df cleaned. shape:"+str(X_df.shape)+"\n")


    print("X_df=\n",X_df[400:460])
   
    X=X_df.to_numpy()
  
##############################################################    

    print("y=\n",y[400:440])

    print("X shape:",X.shape)
    f.write("X shape:"+str(X.shape)+"\n")
    print("y shape:",y.shape)
    f.write("y shape:"+str(y.shape)+"\n")

###########################################################

   

    print("Train,Test split and scaling.")
    f.write("Train,Test split and scaling.\n")


    X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

    scaler=StandardScaler()
    scaler.fit(X_train)

    joblib.dump(scaler,open(scaler_save,"wb"))
    print("Scaler saved to:",scaler_save)
    f.write("Scaler saved to:"+str(scaler_save)+"\n")

    
    X_scaled_train=scaler.transform(X_train)
    X_scaled_test=scaler.transform(X_test)

    
##    regressor = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, learning_rate='constant', eta0=0.01, max_iter=1000)
##    regressor.fit(X_train, y_train)
##
##    joblib.dump(regressor,open("SGDRegressorNS.p","wb"))
## 
##    predictions = regressor.predict(X_scaled_test)
##    #print(predictions)
##    print("SGDR regressor no scaling score:",regressor.score(X_scaled_test, y_test))

    param_grid = {
        "alpha": [1e-08,1e-07, 1e-06, 1e-05],
        "penalty": [None, "l2"],
        "eta0": [0.001, 0.005, 0.01,0.1],
        "max_iter": [10000, 30000, 50000]
    }

    regressor = SGDRegressor(loss='squared_loss', learning_rate='constant')
    grid_search = GridSearchCV(regressor, param_grid, cv=3, scoring='neg_mean_absolute_error',iid=False)
    grid_search.fit(X_scaled_train, y_train)



##    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
##    title = "Learning Curve (SVC)"
##    # SVC is more expensive so we do a lower number of CV iterations:
##    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
##    estimator = SVC(gamma=0.001)
##    plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),cv=cv, n_jobs=4)
##
##    plt.show()


    joblib.dump(regressor,open(SGDR_save,"wb"))
    print("SGDRegressor saved to:",SGDR_save)
    f.write("SGDRegressor saved to:"+str(SGDR_save)+"\n")

    print("SGDR best params",grid_search.best_params_)
    f.write("SGDR best params:"+str(grid_search.best_params_)+"\n")

    regressor_best = grid_search.best_estimator_
   # regressor_best.score(X_test, y_test)

    predictions = regressor_best.predict(X_scaled_test)

    print("SGDR MSE=",mean_squared_error(y_test, predictions))
    print("SDGR MAE=",mean_absolute_error(y_test, predictions))
    print("SDGR R2=",r2_score(y_test, predictions))
    print("SDGR predictions=\n",predictions[:10])

    f.write("SGDR MSE="+str(mean_squared_error(y_test, predictions))+"\n")
    f.write("SDGR MAE="+str(mean_absolute_error(y_test, predictions))+"\n")
    f.write("SDGR R2="+str(r2_score(y_test, predictions))+"\n")
    f.write("SDGR predictions=\n"+str(predictions[:10].tolist())+"\n")

    if True:
        pass
    else:


        param_grid = {
       #     "alpha": [1e-07, 1e-06, 1e-05],
       #     "penalty": [None, "l2"],
       #     "eta0": [0.001, 0.005, 0.01],
       #     "max_iter": [3000, 10000, 30000]
             "C":(0.1,1,10,100,1000,2000)
        }
        regressor = LinearSVR()
        grid_search = GridSearchCV(regressor, param_grid, cv=5,scoring='neg_mean_absolute_error',iid=False)
        grid_search.fit(X_scaled_train, y_train)

        joblib.dump(regressor,open(LSVR_save,"wb"))
        print("LinearSVR saved to:",LSVR_save)
        f.write("LinearSVR saved to:"+str(LSVR_save)+"\n")


        print("Linear SVR best params",grid_search.best_params_)
        f.write("Linear SVR best params"+str(grid_search.best_params_)+"\n")

        regressor_best = grid_search.best_estimator_
       # regressor_best.score(X_test, y_test)

        predictions = regressor_best.predict(X_scaled_test)

        print("L SVR MSE=",mean_squared_error(y_test, predictions))
        print("L SVR MAE=",mean_absolute_error(y_test, predictions))
        print("L SVR R2=",r2_score(y_test, predictions))
        print("L SVR predictions=\n",predictions[:10])
        
        f.write("L SVR MSE="+str(mean_squared_error(y_test, predictions))+"\n")
        f.write("L SVR MAE="+str(mean_absolute_error(y_test, predictions))+"\n")
        f.write("L SVR R2="+str(r2_score(y_test, predictions))+"\n")
        f.write("L SVR predictions=\n"+str(predictions[:10].tolist())+"\n")


        param_grid = {
       #     "alpha": [1e-07, 1e-06, 1e-05],
       #     "penalty": [None, "l2"],
       #     "eta0": [0.001, 0.005, 0.01],
       #     "max_iter": [3000, 10000, 30000]
        #     "kernel":("linear"),
             "epsilon":(0.01,0.02),
             "C":(10,100,300)
        }
        regressor = SVR(kernel="linear")
        grid_search = GridSearchCV(regressor, param_grid, cv=5,scoring='neg_mean_absolute_error')
        grid_search.fit(X_scaled_train, y_train)

        joblib.dump(regressor,open(SVR_save,"wb"))
        # regressor=joblib.load(open("SGDRegressor.p","rb"))
        print("SVR saved to:",SVR_save)
        f.write("SVR saved to:"+str(SVR_save)+"\n")


        print("SVR best params",grid_search.best_params_)
        f.write("SVR best params"+str(grid_search.best_params_)+"\n")

        regressor_best = grid_search.best_estimator_
       # regressor_best.score(X_test, y_test)

        predictions = regressor_best.predict(X_scaled_test)

        print("SVR MSE=",mean_squared_error(y_test, predictions))
        print("SVR MAE=",mean_absolute_error(y_test, predictions))
        print("SVR R2=",r2_score(y_test, predictions))
        print("SVR predictions=\n",predictions[:10])

        f.write("SVR MSE="+str(mean_squared_error(y_test, predictions))+"\n")
        f.write("SVR MAE="+str(mean_absolute_error(y_test, predictions))+"\n")
        f.write("SVR R2="+str(r2_score(y_test, predictions))+"\n")
        f.write("SVR predictions=\n"+str(predictions[:10].tolist())+"\n")


    param_grid = {
   #     "alpha": [1e-07, 1e-06, 1e-05],
   #     "penalty": [None, "l2"],
   #     "eta0": [0.001, 0.005, 0.01],
   #     "max_iter": [3000, 10000, 30000]
    #     "kernel":("linear"),
         "max_depth":[20,30,40],
         "min_samples_split":[3,5]
    }
    regressor = RandomForestRegressor(n_estimators=1000,random_state=42,oob_score=True,bootstrap=True)
    grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error',verbose=True,n_jobs=-1)
    grid_search.fit(X_train, y_train)

    joblib.dump(regressor,open(RFR_save,"wb"))
    print("SVR saved to:",RFR_save)
    f.write("SVR saved to:"+str(RFR_save)+"\n")


    print("Random Forest best params",grid_search.best_params_)
    f.write("Random Forest best params"+str(grid_search.best_params_)+"\n")

    regressor_best = grid_search.best_estimator_
    print("RF best score:",regressor_best.score(X_test, y_test))
    f.write("RF best score:"+str(regressor_best.score(X_test, y_test))+"\n")

    predictions = regressor_best.predict(X_test)

    print("RF MSE=",mean_squared_error(y_test, predictions))
    print("RF MAE=",mean_absolute_error(y_test, predictions))
    print("RF OOB score=",regressor_best.oob_score_)
    print("RF OOB prediction=",regressor_best.oob_prediction_)
    print("RF R2=",r2_score(y_test, predictions))
    print("RF predictions=\n",predictions[:10])

    f.write("RF MSE="+str(mean_squared_error(y_test, predictions))+"\n")
    f.write("RF MAE="+str(mean_absolute_error(y_test, predictions))+"\n")
    f.write("RF OOB score="+str(regressor_best.oob_score_)+"\n")
    f.write("RF OOB prediction="+str(regressor_best.oob_prediction_)+"\n")
    f.write("RF R2="+str(r2_score(y_test, predictions))+"\n")
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
    cols=X_df[X_df.columns[feature_sorted]].columns
    print(cols)
    for name,score in zip(X_df.columns, regressor_best.feature_importances_):
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


  #  print("X_test=\n",X_test[:10,:],"y_predict:",predictions[:10].reshape(-1,1))
    j=pd.DataFrame(X_test,columns=["date_encode","productgroup","prod_encode","previous_qty","day_delta","week_delta","two_week_delta","month_delta","day_of_year","week_of_year","month_of_year","year","day_order_delta","week_order_delta","two_week_order_delta","month_order_delta"])

 #  print("j=\n",j)

    j["date"] = j.date_encode.astype(int).map(dt.datetime.fromordinal)


    encoder=joblib.load(open(product_encode_save,"rb"))
    j["product"]=encoder.inverse_transform(j["prod_encode"].astype(int).to_numpy())

    j=pd.DataFrame(np.hstack((X_test,predictions.reshape(-1,1))))     #,columns=["date_encode","productgroup","prod_encode","previous_qty","weekly_previous_qty","two_weekly_previous_qty","monthly_previous_qty","day_delta","day_of_year","week_of_year","month_of_year","year","day_order_delta","last_month_order_qty","last_week_order_qty","last_two_week_order_qty","predict_qty"])

  
 #   j["date"] = pd.Timestamp(dt.datetime.today()-j.day_delta.min()+j.day_delta)

#  dt.datetime.today().strftime("%d/%m/%Y")
 #   LY_product_encode_save=joblib.load(open("LY_product_encode_save.p","rb"))
 #   encoder2=joblib.load(open(glset_encode_save,"rb"))
 #   j["glset"]=encoder2.inverse_transform(j["glset_encode"].astype(int).to_numpy())

    print("j=\n",j[0:15].to_string())
    

    j= j[["product","date","predict_qty"]]  
    j.sort_values(by=["date"],axis=0,ascending=True,inplace=True)
    print("Sales Qty Predictions=\n",j[0:15].to_string())
   

    
    f.close()
    return

    

if __name__ == '__main__':
    main()
