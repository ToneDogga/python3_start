import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import timeit

df=pd.read_excel("salestransslice1.xlsx", "Sheet1",header=0,convert_float=True)
df['day_delta'] = (df.date.min()-df.date).dt.days
df.fillna(0,inplace=True)
#df2=df.loc[(df['productgroup'] >= 10) & (df['productgroup'] <= 14)]
df=df[(df['productgroup'] >= 10) & (df['productgroup'] <= 14)]

#print("new df=\n",df2)
df.cat = df.cat.astype(int)
df.loc[:,"qty_ctns"] = round(df.qty/8,0).astype(int)
df.sort_values(by=['code','day_delta', 'productgroup','product'],ascending=[True,False,True,True],inplace=True)

#print(df)
X=df.loc[:,["day_delta","cat","code","productgroup","product"]]
#Y=df.iloc[0:1000,-1].values
#Y=df.iloc[:,-1].values
#Y=df2.loc[:,["qty","qty_ctns"]]
#leny=len(Y)
#Y=df.loc[:,["qty_ctns"]].reshape((len(df),))
Y=df["qty_ctns"]



#Y2=np.reshape(Y,leny)
print(Y)

print("df2 shpae=",df.shape)

print("x shape=",X.shape)
print("ÿ shape=",Y.shape)
#print("X=\n",X)
#print("Y=\n",Y)   #.to_string())


#print("Y class balance:",Counter(Y))
#print("Y class set:",set(Y))
#df3 = pd.crosstab(df2['product'], df2['qty_ctns'])   #['product'])
#df3 = pd.crosstab(df2['day_delta'],pd.crosstab(df2['product'], df2['qty_ctns']))   #['product'])
#df3=pd.crosstab(df2, [df2, df2], rownames=['day_delta'], colnames=['product', 'qty_ctns'])
#df3=pd.crosstab([df2['code'],df2['qty_ctns']],[df2['product'],df2['day_delta']] ,rownames=['code','qty_ctns'], colnames=['product', 'day_delta'])

#print(df3)




X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=42)

X_train_dict=pd.DataFrame(X_train).to_dict("records")   
X_test_dict=pd.DataFrame(X_test).to_dict("records")   


#print("x train dict[0]=",X_train_dict[0])


dict_one_hot_encoder=DictVectorizer(sparse=False,dtype=int)


X_train=dict_one_hot_encoder.fit_transform(X_train_dict)
X_test=dict_one_hot_encoder.transform(X_test_dict)

print("len Xtrain after encoding",len(X_train))
print(X_train[0])



svm=SVC(kernel='linear',C=1.0, random_state=42)

svm.fit(X_train,y_train)
SVC(C=1.0,cache_size=200,class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma="auto", kernel="linear", max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001, verbose=False)
##
##accuracy=svm.score(term_docs_test, label_test)
##print("the accuracy on the testing set is: {0:.1f}%".format(accuracy*100))

accuracy=svm.score(X_test,y_test)
print("The accuracy on testing set is: {0:.1f}%".format(accuracy*100))


parameters={ "C": (0.1,0.5,1,10,100)}
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
