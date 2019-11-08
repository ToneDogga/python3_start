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

df=pd.read_excel("salestransslice1.xlsx", "Sheet1",header=0,convert_float=True)
df['day_delta'] = (df.date.min()-df.date).dt.days
df.fillna(0,inplace=True)
df2=df.loc[(df['productgroup'] >= 10) & (df['productgroup'] <= 14)]
print("new df=\n",df2)
df2.cat = df2.cat.astype(int)
df2["qty_ctns"] = round(df2.qty/8,0).astype(int)
df2.sort_values(by=['code','day_delta', 'productgroup','product'],ascending=[True,False,True,True],inplace=True)

#print(df)
X=df2.loc[:,["day_delta","cat","code","productgroup","product"]]
#Y=df.iloc[0:1000,-1].values
#Y=df.iloc[:,-1].values
#Y=df2.loc[:,["qty","qty_ctns"]]
Y=df2.loc[:,["qty_ctns"]]

print("X=\n",X)
print("Y=\n",Y)   #.to_string())


print("Y class balance:",Counter(Y))
print("Y class set:",set(Y))
df3 = pd.crosstab(df2['qty_ctns'], df2['day_delta'])   #['product'])
print(df3)




X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=42)

X_train_dict=pd.DataFrame(X_train).to_dict("records")   
X_test_dict=pd.DataFrame(X_test).to_dict("records")   


print("x train dict[0]=",X_train_dict[0])


dict_one_hot_encoder=DictVectorizer(sparse=False,dtype=int)


X_train=dict_one_hot_encoder.fit_transform(X_train_dict)
X_test=dict_one_hot_encoder.transform(X_test_dict)

print("len Xtrain after encoding",len(X_train))
print(X_train[0])

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
