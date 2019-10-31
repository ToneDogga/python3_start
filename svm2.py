from sklearn.datasets import fetch_20newsgroups
import glob
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
from nltk.corpus import names
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.svm import SVC


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

categories=["alt.atheism","talk.religion.misc","comp.graphics","sci.space","rec.sport.hockey"]
data_train=fetch_20newsgroups(subset="train",categories=categories, random_state=42)
data_test=fetch_20newsgroups(subset="test",categories=categories, random_state=42)

cleaned_train=clean_text(data_train.data)
label_train=data_train.target
print("label train=",label_train)
cleaned_test=clean_text(data_test.data)
label_test=data_test.target

print(len(label_train),len(label_test))
print(Counter(label_train))
print(Counter(label_test))

tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True,max_df=0.5,stop_words='english',max_features=8000)
term_docs_train=tfidf_vectorizer.fit_transform(cleaned_train)
term_docs_test=tfidf_vectorizer.transform(cleaned_test)

svm=SVC(kernel='linear',C=1.0,random_state=42)

svm.fit(term_docs_train,label_train)

SVC(C=1.0,cache_size=200,class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma="auto", kernel="linear", max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001, verbose=False)

accuracy=svm.score(term_docs_test, label_test)
print("the accuracy on the testing set is: {0:.1f}%".format(accuracy*100))

prediction=svm.predict(term_docs_test)
report=classification_report(label_test,prediction)
print(report)


##
##
##emails=[]
##labels =[]
##file_path="enron-spam/enron1/spam"
##for filename in glob.glob(os.path.join(file_path,"*.txt")):
##    with open(filename,"r",encoding="ISO-8859-1") as infile:
##        emails.append(infile.read())
##        labels.append(1)
##
##file_path="enron-spam/enron1/ham"
##for filename in glob.glob(os.path.join(file_path,"*.txt")):
##    with open(filename,"r",encoding="ISO-8859-1") as infile:
##        emails.append(infile.read())
##        labels.append(0)
##


#print("labels=\n",labels)
##
##k=10
##k_fold=StratifiedKFold(n_splits=k)
##
## 
##all_names=set(names.words())
##lemmatizer=WordNetLemmatizer()
##
##cleaned_emails=clean_text(emails)
##cleaned_emails_np=np.array(cleaned_emails)
##labels_np=np.array(labels)
##
##smoothing_factor_option=[1.0,2.0,3.0,4.0,5.0]
##
##
##auc_record=defaultdict(float)
##for train_indices, test_indices in k_fold.split(cleaned_emails,labels):
##    X_train, X_test = cleaned_emails_np[train_indices], cleaned_emails_np[test_indices]
##    Y_train, Y_test = labels_np[train_indices], labels_np[test_indices]
##    tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True,max_df=0.5,stop_words='english',max_features=8000)
##    term_docs_train=tfidf_vectorizer.fit_transform(X_train)
##    term_docs_test=tfidf_vectorizer.transform(X_test)
##    for smoothing_factor in smoothing_factor_option:
##        clf=MultinomialNB(alpha=smoothing_factor,fit_prior=True)
##        clf.fit(term_docs_train,Y_train)
##        prediction_prob=clf.predict_proba(term_docs_test)
##        pos_prob=prediction_prob[:,1]
##        auc=roc_auc_score(Y_test,pos_prob)
##        auc_record[smoothing_factor]+=auc
##print("max features smoothing   fit prior   auc")
##for smoothing, smoothing_record in auc_record.items():
##    print("     8000     {0}    true   {1:.4f}".format(smoothing,smoothing_record/k))
##    
##
##





