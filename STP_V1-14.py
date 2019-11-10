# SVC solution to finding patterns to forecast on using salestrans written by Anthony Paech 9/11/19
# probs
# currently limited by memory issues
# the one hot encoder creates lots of extra dimension to encode categories like code and product
#  how
from __future__ import print_function
from __future__ import division

from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn import preprocessing #import LabelBinarizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from array import *
from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#from sklearn.decomposition import NMF

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
#import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
#from sklearn.datasets import fetch_20newsgroups

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
#from collections import Counter
from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV
import timeit
import sys


# Save Model Using joblib
import pandas
from sklearn import model_selection
#from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
import joblib

def save_model(model,filename):
##    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
##    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
##    dataframe = pandas.read_csv(url, names=names)
##    array = dataframe.values
##    X = array[:,0:8]
##    Y = array[:,8]
##    test_size = 0.33
##    seed = 7
##    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
##    # Fit the model on training set
##    model = LogisticRegression()
##    model.fit(X_train, Y_train)
    # save the model to disk
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

def discrete_buckets(array,buckets):
    # divide the classes into a bucket number of equally sized ranges
    a=pd.cut(array,buckets,labels=range(len(buckets)-1),right=False,retbins=False)
    return a

def import_clean_bin_sheet(filename,customer_code):
    #df=pd.read_excel(sys.argv[1], "Sheet1",header=0,nrows=sys.argv[2],convert_float=True)
    df=pd.read_excel(sys.argv[1], "Sheet1",dtype={"cat":np.int32,"productgroup":np.int32,"qty":np.int32},convert_float=True).head(int(sys.argv[3]))
    print(filename,"import rows=",len(df))
    
    #df=pd.read_excel("salestransslice2.xlsx", "Sheet1",header=0,convert_float=True)
   # df['day_delta'] = (df.date.max()-df.date).dt.days
    #df['day_delta'] = (df.date-df.date.min()).dt.days
    #print(len(df))
    df.fillna(0,inplace=True)
    #df=df[df.productgroup.apply(lambda x: x.isnumeric())]
    #df=df[df.qty.apply(lambda x: x.isnumeric())]
    #print(len(df))
    df=df[pd.to_numeric(df['qty'], errors='coerce').notnull()]
    #print(len(df))
    df=df[pd.to_numeric(df['productgroup'], errors='coerce').notnull()]
    #print(len(df))
    df=df[pd.to_numeric(df['cat'], errors='coerce').notnull()]

    #print("afrer to numeric",df)
    #print(len(df))
    before=len(df)
    #df2=df.loc[(df['productgroup'] >= 10) & (df['productgroup'] <= 14)]
 #   df["cat"] = df.cat.astype(int)
 #   df["qty"]=df.qty.astype(int)
 #   df["productgroup"]=df.productgroup.astype(int)
    #print("before cat",len(df))
    df['day_delta'] = (df.date.max()-df.date).dt.days
    df=df[df['cat']!=86]
    #print("after cat",len(df))
    df=df[df['code']!="CASHSHOP"]
    df=df[((df['productgroup'] >= 10) & (df['productgroup'] <= 14))]
   # df=df[(df['code']!="CASHSHOP") & (df['productgroup'] >= 10) & (df['productgroup'] <= 14) & (df["qty"]>=8) & (df["qty"]<=1120)]
    df=df[(df["qty"]>=8) & (df["qty"]<=1120)]
    df=df[df['code']==customer_code]
 #   df['day_delta'] = (df.date.max()-df.date).dt.days

    #print(len(df))
    after=len(df)
    print("Import, clean and bin",before,"rows of",filename,"for customer",customer_code,"leaves remaining rows",after,"({0:.2f}%)".format(after/before*100))

    #print("\n\ndf.head(40)=\n",df.head(40))

    df.loc[:,"qty_ctns"] = round(df.qty/8,0).astype(int)
    #df.sort_values(by=['code','day_delta', 'productgroup','product'],ascending=[True,False,True,True],inplace=True)

    #print(df)
    X=df.loc[:,["day_delta","productgroup","product"]]   #.astype([np.int32,np.int32,np.str,np.int32,np.str])
    #X=df.loc[:,["day_delta","cat","code","productgroup","product"]]

    #Y=df.iloc[0:1000,-1].values
    #Y=df.iloc[:,-1].values
    #Y=df2.loc[:,["qty","qty_ctns"]]
    #leny=len(Y)
    #Y=df.loc[:,["qty_ctns"]].reshape((len(df),))
    Y=df["qty_ctns"].astype(int)
    buckets=[1,2,4,8,16,40,80,2000]
    #print("before bucketing Y=",Y.head(20))
    y=discrete_buckets(Y,buckets)
    #print("buckets=",buckets,"after bucketing Y=",y.head(20))
    return X,y



def main():
    
    if(len(sys.argv) < 3 ) :
        print("Usage : python STP_v1-xx.py import_spreadsheet_name.xlsx customer_code number_of_rows")
        sys.exit()

    print("salestrans analysis starting....")
    customer_code=sys.argv[2]
    X,y=import_clean_bin_sheet(sys.argv[1],sys.argv[2])

    print("X=\n",X)
   # print("y=\n",y)
    #Y2=np.reshape(y,leny)
    #print(y)

    #print("df2 shape=",df.shape)
    print("customer code:",customer_code)
    print("x shape=",X.shape)
    print("y shape=",y.shape)
    print("Frequency of carton counts in transactions:",Counter(y))
    #print("X=\n",X)
    #print("y=\n",y)   #.to_string())



    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=42)

    X_train_dict=pd.DataFrame(X_train).to_dict("records")   
    X_test_dict=pd.DataFrame(X_test).to_dict("records")   


    #print("x train dict[0]=",X_train_dict[0])

    # memory friendly alternative
    #vectorizer = HashingVectorizer(n_features=2**20)   # 2**4
    #X_train = vectorizer.fit_transform(X_train)
    #X_test= vectorizer.transform(X_test)



    dict_one_hot_encoder=DictVectorizer(sparse=False,dtype=int)
    X_train=dict_one_hot_encoder.fit_transform(X_train_dict)
    X_test=dict_one_hot_encoder.transform(X_test_dict)

#   lb=preprocessing.LabelBinarizer()


    #print("len Xtrain after encoding",len(X_train))
    #print("X_train shape:",X_train.shape)
    #print("X_test shape:",X_test.shape)



    svm=SVC(kernel='linear',C=1, random_state=42)
  #  svm=SVC(kernel='linear', random_state=42)

    svm.fit(X_train,y_train)
    SVC(C=1.0,cache_size=200,class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma="auto", kernel="linear", max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001, verbose=False)
    ##
    ##accuracy=svm.score(term_docs_test, label_test)
    ##print("the accuracy on the testing set is: {0:.1f}%".format(accuracy*100))

    #accuracy=svm.score(X_test,y_test)
    #print("The accuracy on testing set is: {0:.1f}%".format(accuracy*100))


    parameters={ "C": (0.1,0.5,1,5,10)}
    grid_search=GridSearchCV(svm, parameters, n_jobs=-1,cv=3)

    start_time=timeit.default_timer()
    grid_search.fit(X_train, y_train)
    print("--- %0.1fs seconds -----" % ( timeit.default_timer()-start_time))

    print(grid_search.best_params_)
    print(grid_search.best_score_)

    svc_libsvm_best=grid_search.best_estimator_
    accuracy=svc_libsvm_best.score(X_test, y_test)

    print("The accuracy on testing set is: {0:.1f}%".format(accuracy*100))

    prediction=svm.predict(X_test)
    report=classification_report(y_test,prediction)
    print(report)


    filename = 'finalized_model.sav'
    save_model(svc_libsvm_best,filename)
    print("model saved")
    loaded_model=load_model(filename)
    print("model loaded")
    result = loaded_model.score(X_test, y_test)
   # print(result)
    print("The accuracy on loaded testing set is: {0:.1f}%".format(result*100))

    #print("Y class balance:",Counter(Y))
    #print("Y class set:",set(Y))
    #df3 = pd.crosstab(df2['product'], df2['qty_ctns'])   #['product'])
    #df3 = pd.crosstab(df2['day_delta'],pd.crosstab(df2['product'], df2['qty_ctns']))   #['product'])
    #df3=pd.crosstab(df2, [df2, df2], rownames=['day_delta'], colnames=['product', 'qty_ctns'])
    #df3=pd.crosstab([df2['code'],df2['qty_ctns']],[df2['product'],df2['day_delta']] ,rownames=['code','qty_ctns'], colnames=['product', 'day_delta'])

    #print(df3)


    products=X["product"].tolist()
    #productgroups=X["productgroup"]).tolist()
    
   # productgroups=array_str_list(X["productgroup"])     # copy to a list of strings
    productgroups=map(str, X["productgroup"].tolist())
    days=map(str, X["day_delta"].tolist())

    print("product groups=\n",products,"days=",days)

    cv=CountVectorizer(stop_words="english",max_features=500)
##    transformed=cv.fit_transform(products)
##    nmf=NMF(n_components=100,random_state=43).fit(transformed)
##    for topic_idx, topic in enumerate(nmf.components_):
##        label="{}: ".format(topic_idx)
##        print(label," ".join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))




    transformed=cv.fit_transform(products)

    km=KMeans(n_clusters=5)
    km.fit(transformed)
    labels=X["day_delta"]
    plt.scatter(km.labels_,labels)
    plt.xlabel("Products")
    plt.ylabel("Day Delta")
    plt.show()


    #parameters={"max_depth":[3,5,7,9,11,None]}
    ##
    ##random_forest=RandomForestClassifier(n_estimators=100,criterion="gini",min_samples_split=30,n_jobs=-1)
    ##grid_search=GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring="roc_auc")
    ##grid_search.fit(X_train,Y_train)
    ##print("grid search best params=",grid_search.best_params_)
    ##
    ##random_forest_best=grid_search.best_estimator_
    ##pos_prob=random_forest_best.predict_proba(X_test)[:,1]
    ##
    ##print("the ROC AUC on testing set is {0:.3f}".format(roc_auc_score(Y_test, pos_prob)))

    return


###########################################


 

if __name__ == '__main__':
    main()

