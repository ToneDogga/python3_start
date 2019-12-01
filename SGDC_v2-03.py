# SGDC Stochastic gradient descent classifier
# written by Anthony Paech  29/11/19
#
# Take a excel spreadsheet that is a slice of the salestrans
# ideally, filtered by customer code and product group and saved as a CSV
# such that the features are simply
# "productcode" "day_delta"
# and the classes are binned values of the "qty"
# these are read into a dictionary (the DictVectoriser is forgiving and very good)
# and vectorised with one hot encoder
#  also test the accuracy with other models like naive bayes, random forest and Support vector machine
#
# pseudocode
# load slice of sales trans straight from excel
# choose columns X,y
# cleanup NaN's
# convert date to day_delta
# create order_day_delta (the day gap between the last order and this one)
# split into X_train, X_test, y_train, y_test
# convert X_train and X_test to dictionaries
# convert y_train, y_test to np.array()
# bin the Y_train, Y_test data
# use DictVectorizer on X_train
# use SGDClassifier object
# run using GridSearch
# use ROC AUC to score predictions on y_test based on X_test
#

#!/usr/bin/python3
from __future__ import print_function
from __future__ import division



import numpy as np
import pandas as pd
import csv
import datetime as dt
import joblib
import pickle

import timeit
    
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline


def save_model(model,filename):

    #filename = 'finalized_model.sav'
    joblib.dump(model, filename)
    return 

def load_model(filename):
    # some time later...

    # load the model from disk
    loaded_model = joblib.load(filename)
 #   result = loaded_model.score(X_test, Y_test)
 #   print(result)
    return loaded_model





##def week_diff(start, end):
##    x = pd.to_datetime(end) - pd.to_datetime(start)
##    return int(x / np.timedelta64(1, 'W'))

def week_delta(df):

    df['week_delta'] = (df.date.max()-df.date).dt.days.astype(int)/7
 #   df['day_delta'] = (df.date.max()-df.date).dt.days.astype(int)

 #   maxdate=df.day_delta.max().astype(int)
 #   df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.weekofyear

#    df['day_delta'] = (df.date-df.date.min()).dt.days
 #   df.drop(columns='date',axis=1,inplace=True)
 #   df.drop(df['date'],inplace=True)
##    date_N_days_ago = datetime.now() - timedelta(days=2)
##
##    print(datetime.now())
##    print(date_N_days_ago)
    return df


def order_delta(df):
    df.sort_values(by=["product","week_delta"],ascending=[True,False],inplace=True)
    df["order_delta"]=round(df.week_delta.diff(periods=-1),0)
    df.order_delta.clip(lower=0,inplace=True)
 #   print(df.head(60))
    return df





def remove_non_trans1(df):
 #  df.drop(df[df.score < 50].index, inplace=True)
    #df.drop(df[df.qty<0.0].index,inplace=True)
    df["dd"]=(df.qty<=0.0)  #  | df.order_delta==0.0)
    df.drop(df[df.dd==True].index,inplace=True)
    df.drop(['dd'], axis=1,inplace=True)
    return df


def remove_non_trans2(df):
 #  df.drop(df[df.score < 50].index, inplace=True)
    #df.drop(df[df.qty<0.0].index,inplace=True)
    df["dd"]=(df.order_delta==0.0)
    df.drop(df[df.dd==True].index,inplace=True)
    df.drop(['dd'], axis=1,inplace=True)
    return df




def bin_y(df,bins):
    df['qty_bins']=pd.cut(df['qty'].values.astype(float),bins,labels=range(len(bins)-1),right=True,retbins=False)
    return df


##def read_in_csvdata(filename,n, offset=0):
##    X_dict, y = [], []
##    with open(filename, 'r') as csvfile:
##        reader = csv.DictReader(csvfile)
##        for i in range(offset):
##            next(reader)
##        i = 0
##        for row in reader:
##            i += 1
##            y.append(int(row['qty']))
##            del row['qty'] #, row['location'], row['code'], row['refer'], row['linenumber']
##            X_dict.append(row)
##            if i >= n:
##                break
##    return X_dict, y


def read_in_exceldata(filename,n):   #,offset=0):
    df=pd.read_excel(filename, "Sheet1")
    X=df.iloc[0:n,:2]
    Y=df.iloc[0:n,4].values
#    print(X)
    X=day_delta(X)
  #  print("X=\n",X)
    #print(Counter(Y))
  #  X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)
  #  print("x train=",X_train[:10])
  #  X_dict_train=dict(X_train)
    #d = dict(enumerate(myarray.flatten(), 1))
  #  X_dict_test=dict(X_test)
    return X_dict_train,X_dict_test,y_train,y_test


def read_excel(filename,rows):
    xls = pd.ExcelFile(filename)    #'salestransslice1.xlsx')
    if rows==-1:
        return xls.parse(xls.sheet_names[0])
    else:        
        return xls.parse(xls.sheet_names[0]).head(rows)


def write_excel(df,outfilename):
    df.to_excel(outfilename)    #'salestransslice1.xlsx')
    return
 
def write_csv(df,outfilename):
    df.to_csv(outfilename,index=False)    #'salestransslice1.xlsx')
    return
 


##def discrete_ybins(array,bins):
##    # divide the classes (or features) into a bucket number of equally sized ranges
##    return pd.cut(array.astype(float),bins,labels=range(len(bins)-1),right=True,retbins=False)  #.astype(int)
##     


def main():
    bins=[0,15,31,63,127,10000]
    infile="salestransslice4-FLNOR.xlsx"
    outfile="salestransslice4-FLNOR.csv"
    df=read_excel(infile,-1)
    #print(df.head(10))
    #print("b=",df.shape)
    df=remove_non_trans1(df)

    #print(df.head(10))
    #print("c=",df.shape)
    df=week_delta(df)
    #df=order_delta(df,maxday)
    df=order_delta(df)
    df=remove_non_trans2(df)


    df=bin_y(df,bins)
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    #df.reindex()
    print("df=\n",df.head(90)) #.to_string())
    print("d=",df.shape)

    write_csv(df,outfile)
    df.drop(columns=["date","week_delta","qty"],inplace=True)


    label_encoder=LabelEncoder()
    df["prod_encode"] = label_encoder.fit_transform(df["product"].to_numpy())
    #X_test = label_encoder.transform(test)
    df.drop(columns="product",inplace=True)
    df = df[["prod_encode","week_of_year","order_delta","qty_bins"]]
    print("df=\n",df.tail(80))

    X=df.iloc[:,0:3].to_numpy()
    y=df.iloc[:,3].to_numpy()



    print("X=",X[0:30])
    print("y=",y[0:30])



    X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.33,random_state=42)

    #print("X_train=",X_train[0:30])
    #print("X_test=",X_test[0:30])
    #print("y_train=",y_train[0:30])
    #print("y_test=",y_test[0:30])



    # combined with grid search

    parameters = {'penalty': ['l2', None],
                  'alpha': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02],
                  'eta0': [0.01, 0.1, 1, 10, 100,1000]}
    print("\n\nGrid search using Stochastic Gradient Decent with parameters:",parameters)
    sgd_lr = SGDClassifier(loss='log', learning_rate='constant', eta0=0.01, fit_intercept=True, max_iter=5000)

    grid_search = GridSearchCV(sgd_lr, parameters, n_jobs=-1, cv=3)

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    sgd_lr_best = grid_search.best_estimator_
    accuracy = sgd_lr_best.score(X_test, y_test)
    print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))

    filename = 'finalized_SGD_model.sav'
    print("save model to",filename)
    save_model(sgd_lr,filename)

    #####
    print("\n\n\nload model from",filename)

    testmod=load_model(filename)
    grid_search = GridSearchCV(testmod, parameters, n_jobs=-1, cv=3)

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    testmod_best = grid_search.best_estimator_
    accuracy = testmod_best.score(X_test, y_test)
    print('The accuracy loaded model on testing set is: {0:.1f}%'.format(accuracy*100))


#######################################

    print("\n\nSupport Vector machine classification...")

    svm=SVC(kernel='linear',C=1.0,random_state=42)

    svm.fit(X_train,y_train)

    #SVC(C=1.0,cache_size=200,class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma="auto", kernel="linear", max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001, verbose=False)
    filename = 'finalized_SVM_model.sav'
    print("save model to",filename)
    save_model(svm,filename)

    #####
    print("\n\n\nload SVM model from",filename)
    testmod=load_model(filename)

    

    accuracy=testmod.score(X_test, y_test)
    print("the accuracy on the testing set is: {0:.1f}%".format(accuracy*100))


    prediction=testmod.predict(X_test)
    report=classification_report(y_test,prediction)
    print(report)
    
    pipeline = Pipeline([
       # ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svc', LinearSVC()),
    ])

    parameters_pipeline = {
##        'tfidf__max_df': (0.25, 0.5),
##        'tfidf__max_features': (40000, 50000),
##        'tfidf__sublinear_tf': (True, False),
##        'tfidf__smooth_idf': (True, False),
        'svc__C': (0.1, 1, 10, 100),
    }
##
    grid_search = GridSearchCV(pipeline, parameters_pipeline, n_jobs=-1, cv=3)




##    grid_search = GridSearchCV(testmod, parameters, n_jobs=-1, cv=3)
##
##
##    start_time = timeit.default_timer()
##    grid_search.fit(X_train, y_train)
##    print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))
##
##    print(grid_search.best_params_)
##    print(grid_search.best_score_)
##
##    svc_libsvm_best = grid_search.best_estimator_
##    accuracy = svc_libsvm_best.score(X_test, y_test)
##    print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
##

  #  from sklearn.svm import LinearSVC
 #   svc_linear = LinearSVC()
 #   grid_search = GridSearchCV(svc_linear, parameters, n_jobs=-1, cv=3)

    start_time = timeit.default_timer()
    grid_search.fit(X_train, y_train)
    print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))

    print(grid_search.best_params_)
    print(grid_search.best_score_)
    svc_linear_best = grid_search.best_estimator_
    accuracy = svc_linear_best.score(X_test, y_test)
    print('Linear SVC - The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))




##    pipeline = Pipeline([
##        ('tfidf', TfidfVectorizer(stop_words='english')),
##        ('svc', LinearSVC()),
##    ])
##
##    parameters_pipeline = {
##        'tfidf__max_df': (0.25, 0.5),
##        'tfidf__max_features': (40000, 50000),
##        'tfidf__sublinear_tf': (True, False),
##        'tfidf__smooth_idf': (True, False),
##        'svc__C': (0.1, 1, 10, 100),
##    }
##
##    grid_search = GridSearchCV(pipeline, parameters_pipeline, n_jobs=-1, cv=3)
##
##    start_time = timeit.default_timer()
##    grid_search.fit(X_train, y_train)
##    print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))
##
##    print(grid_search.best_params_)
##    print(grid_search.best_score_)
##    pipeline_best = grid_search.best_estimator_
##    accuracy = pipeline_best.score(X_test, y_test)
##    print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
##



    #prediction_prob=testmod.predict_proba(y_test)
    #pos_prob=prediction_prob[:,1]

    #auc=roc_auc_score(y_test,pos_prob)
    #print("AUC=",auc)
##
##    from sklearn.tree import DecisionTreeClassifier
##    parameters = {'max_depth': [3, 10, None]}
##    decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
##
##    from sklearn.model_selection import GridSearchCV
##    grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
##
##    grid_search.fit(X_train, y_train)
##    print(grid_search.best_params_)
##
##    decision_tree_best = grid_search.best_estimator_
##    pos_prob = decision_tree_best.predict_proba(X_test)[:, 1]
##
##    from sklearn.metrics import roc_auc_score
##    print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test, pos_prob)))
##
##
##
##    from sklearn.ensemble import RandomForestClassifier
##
##    random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
##    grid_search = GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
##    grid_search.fit(X_train, y_train)
##    print(grid_search.best_params_)
##    print(grid_search.best_score_)
##
##    random_forest_best = grid_search.best_estimator_
##    pos_prob = random_forest_best.predict_proba(X_test)[:, 1]
##    print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test, pos_prob)))
##
##
##        

    return



if __name__ == '__main__':
    main()
