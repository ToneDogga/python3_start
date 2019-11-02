from sklearn.datasets import fetch_20newsgroups
import glob
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
from nltk.corpus import names
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import timeit


def letters_only(astr):
    return(astr.isalpha())


def clean_text(docs):
    cleaned_docs=[]
    for doc in docs:
        cleaned_docs.append(" ".join([lemmatizer.lemmatize(word.lower())
                                    for word in doc.split()
                                      if letters_only(word)
                                          and word not in all_names]))
    return(cleaned_docs)



all_names=set(names.words())
lemmatizer=WordNetLemmatizer()
##

#categories=["alt.atheism","rec.sport.hockey","comp.graphics","sci.space","rec.sport.hockey","talk.religion.misc"]
#categories = ["comp.graphics","sci.space"]
categories=None

data_train=fetch_20newsgroups(subset="train",categories=categories, random_state=42)
data_test=fetch_20newsgroups(subset="test",categories=categories, random_state=42)
##
cleaned_train=clean_text(data_train.data)
label_train=data_train.target
##print("label train=",label_train)
cleaned_test=clean_text(data_test.data)
label_test=data_test.target
##



pipeline=Pipeline([("tfidf",TfidfVectorizer(stop_words="english")), ("svc",LinearSVC())])
parameters_pipeline={
    "tfidf__max_df": (0.25,0.5),
    "tfidf__max_features": (40000,50000),
    "tfidf__sublinear_tf":(True, False),
    "tfidf__smooth_idf": (True,False),
    "svc__C": (0.1,1,10,100) }

grid_search=GridSearchCV(pipeline, parameters_pipeline, n_jobs=-1, cv=3)


start_time=timeit.default_timer()
grid_search.fit(cleaned_train, label_train)
print("--- %0.1fs seconds -----" % ( timeit.default_timer()-start_time))

print(grid_search.best_params_)
print(grid_search.best_score_)

pipeline_best=grid_search.best_estimator_
accuracy=pipeline_best.score(cleaned_test, label_test)


print("The accuracy on testing set is: {0:.1f}%".format(accuracy*100))




