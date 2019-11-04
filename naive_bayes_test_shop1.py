import glob
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
##from sklearn.datasets import fetch_20newsgroups
##from nltk.corpus import names
##from nltk.stem import WordNetLemmatizer
##from sklearn.cluster import KMeans
##from sklearn.decomposition import NMF
##
##
import timeit

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score


# bayes theorem simple
# the probability of A happening given B has occured is denoted as P(A|B)
# in bayes theorem this is equal to
#  P(A|B)=  P(B|A).P(A)  /   P(B)
#
# three machines in a factory A, B and C account for 35%, 20% and 45% of the bulb production
# the fraction of defective bulbs produced by each machine is 1.5%, 1% and 2% respectively
# a bulb produced in this factory is deemed defective (D) what are the probabilitys that
# the bulb was produced in each machine A,B or C?
#
#  we know that P(D|A)=0.015, P(D|B)=0.01 and P(D|C)=0.02
#  we also know that P(A)=0.35, P(B)=0.2 and P(C)=0.45
#  we want to find P(A|D), P(B|D) and P(C|D)
#
# so P(A|D)=P(D|A).P(A) / P(D)
# = 0.015 x 0.35 / P(D)
# we know that P(D)=P(D|A).P(A)+P(D|B).P(B)+P(D|C).P(C)
# = 0.015x0.35+0.01x0.2+0.02x0.45
# P(D)=0.00525+0.002+0.009=0.01625
# so P(A|D)=0.323
# P(B|D)=P(D|B).P(B)/(P(D)
# = 0.01 x 0.2 / 0.01625 = 0.123
# P(C|D)=P(D|C).P(C)/P(D)
# = 0.02*0.45 / 0.01625 = 0.5538
#
# (p67)
# if x is a row of a dataset
# it is called a feature of the number of dimensions
# x= {x1,x2,x3,.....xn}
#
# y is the final list of classes that each row can belong to  (usually the right most column)
# if there are k possible classes y1,y2,y3,...,yk
# what is the probability that xn belongs in each class of y?
#
# what is P(yk|x)?
#  P(yk|x) = P(x|yk).P(yk) / P(x)
# P(yk) portrays how classes are distributed.  It is called prior
#
# P(yk|x) is a posterier with the extra knowledge of observation
#
# P(x|yk)=P(x1,x2,x3,....,xn|yk)
# is the joint distribution of n features given the sample belongs to class yk
# that is how likely the features with such values co-occur
# the likeihood
# for this to all work, we assume that the features are independent
#
# posterier =  likelihood * prior / evidence   
#
# so P(x|yk)=P(x1|yk)*P(x2|yk)*P(x3|yk)* ....   *P(xn|yk)  
# this can be learned from a set of training samples
#
# P(x) also called evidence, posterier is proportional to proir  and likelihood
#  Therefore we can kind of ignore the denominator
# so  P(yk|x) = P(x1|yk)*P(x2|yk)*P(x3|yk)* ....   *P(xn|yk)  *   P(yk)
#

##


df=pd.read_excel("shopsales34.xlsx", "shopsales32")
#print(df)
X=df.iloc[0:1996,0:7].values   # features (inputs)
Y= df.iloc[0:1996,8].values    #,np.reshape(   (-1,1))    # classes (output)
#print(X)
#print(Y)
#input("?")
#print("features: X",Counter(X))
print("Class Y:",Counter(Y))


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)



print("X_test=\n",X_test)

#X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)
print("X_train len",len(X_train),"Y_train len",len(Y_train))
print("X_test len",len(X_test),"Y-test len",len(Y_test))

print("X_test shape=",X_test.shape)
print("Y_test shape=",Y_test.shape)
print("X_train shape=",X_train.shape)
print("Y_train shape=",Y_train.shape)

#cleaned_emails=clean_text(X_train)
#print(cleaned_emails[3000])
#term_docs=cv.fit_transform(cleaned_emails)
#term_docs_test=cv.transform(cleaned_test)
#signal_train=cv.fit_transform(X_train)
#output_train=Y_train
#signal_test=cv.fit_transform(X_test)
#output_test=Y_test
clf=MultinomialNB(alpha=1.0,fit_prior=True)
clf.fit(X_train, Y_train)
prediction_prob=clf.predict_proba(X_test)
print(prediction_prob[0:10])
prediction=clf.predict(X_test)
print(prediction[:10])
accuracy=clf.score(X_test, Y_test)
print("the accuracy using MultinomialNB is: {0:.1f}%".format(accuracy*100))
print("confusion matrix=",confusion_matrix(Y_test,prediction,labels=[0,1]))
report=classification_report(Y_test,prediction)
print("\n",report)

pos_prob=prediction_prob[:,1]
##thresholds=np.arange(0.0,1.2,0.1)
##true_pos, false_pos=[0]*len(thresholds), [0]*len(thresholds)
##for pred,y in zip(pos_prob,Y_test):
##    for i, threshold in enumerate(thresholds):
##        if pred >= threshold:
##            # if truth and prediction are both 1
##            if y==1:
##                true_pos[i]+=1
##            # if true is 0 and prediction is 1
##            else:
##                false_pos[i]+=1
##        else:
##            break
##
##
##true_pos_rate=[tp/516.0 for tp in true_pos]
##false_pos_rate=[fp/1191.0 for fp in false_pos]
##
##plt.figure()
##lw=2
##plt.plot(false_pos_rate,true_pos_rate, color='darkorange', lw=lw)
##plt.plot([0,1],[0,1], color='navy',lw=lw,linestyle="--")
##plt.xlim([0.0,1.0])
##plt.ylim([0.0,1.05])
##plt.xlabel('False positive rate')
##plt.ylabel('True positive rate')
##plt.title("receiver operating characteristic")
##plt.legend(loc="lower right")
##plt.show()

print(roc_auc_score(Y_test,pos_prob))
