import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import Vectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

df=pd.read_excel("CTG.xls", "Raw Data")
X=df.iloc[1:2126,3:-2].values
Y=df.iloc[1:2126,-1].values
print(Counter(Y))


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)


##def read_ad_click_data(n, offset=0):
##    X_dict, y = [],[]
##    with open("train","r") as csvfile:
##        reader=csv.DictReader(csvfile)
##        for i in range(offset):
##            reader.next()
##        i=0
##        for row in reader:
##            i+=1
##            y.append(int(row["click"]))
##            del row["click"],row["id"],row["hour"],row["device_id"],row["device_ip"]
##            X_dict.append(row)
##            if i>= n:
##                break
##    return X_dict, y

n_max=10000
#X_dict_train=dict(X_train.flatten())
y_train = Y_train
#print(X_dict_train[0])

#one_hot_encoder=Vectorizer(sparse=False)
dict_one_hot_encoder=DictVectorizer(sparse=False)

X_train=dict_one_hot_encoder.fit_transform(X_train)
print(len(X_train[0]))


#x_dict_test,y_test=read_ad_click_data(n,n)
X_test=dict_one_hot_encoder.transform(X_test)
print(len(X_test[0]))


parameters={"max_depth":[3,10,None]}

##decision_tree=DecisionTreeClassifier(criterion="gini", min_samples_split=30)
##
##grid_search=GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring="roc_auc")
##grid_search.fit(X_train,y_train)
##print("grid search best params=",grid_search.best_params_)
##
##decision_tree_best=grid_search.best_estimator_
##pos_prob=decision_tree_best.predict_proba(X_test)[:,1]
##
##print("the ROC AUC on testing set is {0:.3f}".format(roc_auc_score(y_test, pos_prob)))



random_forest=RandomForestClassifier(n_estimators=100,criterion="gini",min_samples_split=30,n_jobs=-1)
grid_search=GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring="roc_auc")
grid_search.fit(X_train,y_train)
print("grid search best params=",grid_search.best_params_)

random_forest_best=grid_search.best_estimator_
pos_prob=random_forest_best.predict_proba(X_test)[:,1]

print("the ROC AUC on testing set is {0:.3f}".format(roc_auc_score(y_test, pos_prob)))
