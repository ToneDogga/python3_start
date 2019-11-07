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

df=pd.read_excel("shopsales37.xlsx", "shopsales32",header=0,convert_float=True)
X=df.iloc[0:1994,:-2].values
Y=df.iloc[0:1994,-1].values
headings=df.columns.tolist()
print("h=",headings)

print("Y class balance:",Counter(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

X_train_dict=pd.DataFrame(X_train).to_dict("records")   
X_test_dict=pd.DataFrame(X_test).to_dict("records")   

dict_one_hot_encoder=DictVectorizer(sparse=False,dtype=int)
dict_one_hot_encoder.feature_names_=headings

X_train=dict_one_hot_encoder.fit_transform(X_train_dict)
X_test=dict_one_hot_encoder.transform(X_test_dict)

parameters={"max_depth":[3,5,7,9,11,None]}

random_forest=RandomForestClassifier(n_estimators=100,criterion="gini",min_samples_split=30,n_jobs=-1)
grid_search=GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring="roc_auc")
grid_search.fit(X_train,Y_train)
print("grid search best params=",grid_search.best_params_)

random_forest_best=grid_search.best_estimator_
pos_prob=random_forest_best.predict_proba(X_test)[:,1]

print("the ROC AUC on testing set is {0:.3f}".format(roc_auc_score(Y_test, pos_prob)))
