from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import timeit

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

df=pd.read_excel("shopsales32.xlsx", "shopsales32")
#print(df)
X=df.iloc[0:1996,0:7].values
Y=df.iloc[0:1996,8].values
#print(X)
#print(Y)
#input("?")
print("Features:",Counter(Y))


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5,random_state=42)


#svc=SVC(kernel="rbf")
svc=SVC(kernel="linear")
parameters={"C":(100,1e3,1e4,1e5), "gamma":(1e-08,1e-7,1e-6,1e-5)}

grid_search=GridSearchCV(svc, parameters, n_jobs=-1, cv=3)

start_time=timeit.default_timer()
grid_search.fit(X_train,Y_train)
print("%0.3fs" % (timeit.default_timer()-start_time))

print("best params",grid_search.best_params_)
print("best score",grid_search.best_score_)
svc_best=grid_search.best_estimator_
print("best estimator",svc_best)

accuracy=svc_best.score(X_test,Y_test)

print("the accuracy of the testing set is {0:.1f}%".format(accuracy*100))

prediction=svc_best.predict(X_test)
report=classification_report(Y_test, prediction)
print(report)
