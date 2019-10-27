import glob
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

emails=[]
labels =[]
file_path="enron-spam/enron1/spam"
for filename in glob.glob(os.path.join(file_path,"*.txt")):
    with open(filename,"r",encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

file_path="enron-spam/enron1/ham"
for filename in glob.glob(os.path.join(file_path,"*.txt")):
    with open(filename,"r",encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)

#print(len(emails))
#print(len(labels))
#print(emails[7])
#input("?")
      

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
# it is called a feature vector
# x= {x1,x2,x3,.....xn}
#
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







#import nltk
#nltk.download()
#
##from nltk.stem.porter import PorterStemmer
##porter_stemmer=PorterStemmer()
##
##print(porter_stemmer.stem("machines"))
##print(porter_stemmer.stem("learning"))
##
##
##from nltk.stem import WordNetLemmatizer
##lemmatizer=WordNetLemmatizer()
##
##print(lemmatizer.lemmatize("machines"))
##print(lemmatizer.lemmatize("learning"))
##
from sklearn.feature_extraction.text import CountVectorizer
##from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
###from nltk.corpus import shakespeare as sp
##
##import seaborn as sns
##import matplotlib.pyplot as plt
##import numpy as np    
##
from nltk.stem import WordNetLemmatizer
##from sklearn.cluster import KMeans
##from sklearn.decomposition import NMF
##
##
##
##
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
    

def get_label_index(labels):
    from collections import defaultdict
    label_index=defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return(label_index)


def get_prior(label_index):
    # compute prior based on training samples
    #Args:
    #    label_index (grouped sample indicies by class)
    # returns:
    #  a dictionary, with class label as key, corresponding prior as the value    

    prior={label: len(index) for label, index
           in label_index.items()}

    total_count=sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return(prior)


def get_likelihood(term_document_matrix, label_index, smoothing=1):
    # compute likelihood based on training samples
    # args
    #   term_document_matrix (sparse matrix)
    #   label_index (grouped sample indices by class)
    #   smoothing (integer, additive laplace smoothing)
    #
    # Returns:
    #   dictionary, with class as key, corresponding conditional probability P(feature|class)
    #   vector as value

        likelihood={}
        for label, index in label_index.items():
            likelihood[label]=term_document_matrix[index,:].sum(axis=0)+smoothing
            likelihood[label]=np.asarray(likelihood[label])[0]
            total_count=likelihood[label].sum()
            likelihood[label]=likelihood[label] / float(total_count)
        return(likelihood)



def get_posterior(term_document_matrix, prior, likelihood):
# Compute posterier of testing samples, based on prior and likeilhood
# args:
# term_document_matrix (sparse matrix)
# prior (dictionary, with class label as key, corresponding proir as value)
# likeilhood (dictionary, with class label as key, corresponding conditional probability vector as value)
# returns:
# dictionary, with class label as key
# corresponding posterier as value
#
    num_docs=term_document_matrix.shape[0]
    posteriors=[]
    for i in range(num_docs):
        # posterior is proportional to proir * likeihood
        # = exp(log(proir * likeilhood))
        # = exp(log(proir) + log(likelihood))
        posterior = {key:np.log(prior_label)
                     for key, prior_label in prior.items()}
        for label,likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts=term_document_vector.data
            indices=term_document_vector.indices
            for count, index in zip(counts,indices):
                posterior[label]+=np.log(likelihood_label[index])*count
        # exp(-1000):exp(-999) will cause a zero division error,
        # however it equates to exp(0):exp(1)
        min_log_posterior=min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label]-min_log_posterior)
            except:
                # if one's log is exceptionally large, assign it infinity
                posterior[label] = float('inf')
         # normalise it so it all adds up to 1
        sum_posterior=sum(posterior.values())
        for label in posterior:
            if posterior[label]==float('inf'):
                posterior[label]=1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return(posteriors)   
             
    


##
##
###transformed=cv.fit_transform(groups.data)
####print(cv.get_feature_names())
cv=CountVectorizer(stop_words="english",max_features=500)
##groups=fetch_20newsgroups()
##
##
###print(sp.fileids())
###macbeth = sp.words("macbeth.xml")
#print(shakespeare_macbeth)
#print(sp["target_names"])

##print(groups.keys())
##print(groups["target_names"])
##print(groups.target)
##print(np.unique(groups.target))
##print(groups.data[0])
##print(groups.target[0])
##print(groups.target_names[groups.target[0]])
##print(len(groups.data[0]))
##print(len(groups.data[1]))
##sns.distplot(groups.target)
##plt.show()






##sns.distplot(np.log(transformed.toarray().sum(axis=0)))
##plt.xlabel("log count")
##plt.ylabel("frequency")
##plt.title("distribution plot of 500 word sounts")
##plt.show()
##
##cleaned=[]
##
all_names=set(names.words())
###all_names=set(sp.words("macbeth.xml"))
###print(all_names)
lemmatizer=WordNetLemmatizer()
##
##
##for post in groups.data:   #all_names:  #groups.data:
##    cleaned.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))
## #   if letters_only(post):
##  #      cleaned.append(" ".join([lemmatizer.lemmatize(post.lower())]))   # if letters_only(post)]))
##
###print(cleaned)
##transformed=cv.fit_transform(cleaned)
###print(cv.get_feature_names())
###km=KMeans(n_clusters=20)
###km.fit(transformed)
##nmf=NMF(n_components=100,random_state=43).fit(transformed)
##for topic_idx, topic in enumerate(nmf.components_):
##    label="{}: ".format(topic_idx)
##    print(label," ".join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))
##    
####labels=groups.target
####plt.scatter(labels,km.labels_)
####plt.xlabel("Newsgroup")
####plt.ylabel("Cluster")
####plt.show()
##



cleaned_emails=clean_text(emails)
#print(cleaned_emails[3000])
term_docs=cv.fit_transform(cleaned_emails)
#print(term_docs[0])
feature_names=cv.get_feature_names()
#print(feature_names[197])
feature_mapping=cv.vocabulary_
#print(feature_mapping)


label_index=get_label_index(labels)
#print(label_index)
prior=get_prior(label_index)
#print("prior:",prior)

smoothing=1
likelihood=get_likelihood(term_docs, label_index, smoothing)
#print(len(likelihood[0]))

#print("likelihood[0][:5]:",likelihood[0][:5])
      
#print(likelihood[1][:5])

#print("feature names[:5]",feature_names[:5])


#emails_test=[emails[70],emails[8]]
#print("test emails=",emails_test)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)
print("X_train len",len(X_train),"Y_train len",len(Y_train))
print("X_test len",len(X_test),"Y-test len",len(Y_test))

term_docs_train=cv.fit_transform(X_train)
label_index=get_label_index(Y_train)
prior=get_prior(label_index)
likelihood=get_likelihood(term_docs_train,label_index,smoothing)

term_docs_test=cv.transform(X_test)
posterior=get_posterior(term_docs_test,prior,likelihood)

correct=0.0
for pred, actual in zip(posterior, Y_test):
    if actual==1:
        if pred[1]>=0.5:
            correct+=1
        elif pred[0]>0.5:
            correct+=1

print("the accuracy on {0} testing samples is: {1:1f}%".format(len(Y_test),correct/len(Y_test)*100))


cleaned_test=clean_text(emails)
term_docs_test=cv.transform(cleaned_test)
posterior=get_posterior(term_docs_test, prior, likelihood)
#print("spam probabilities:",posterior)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_test, labels, test_size=0.33, random_state=42)
print("X_train len",len(X_train),"Y_train len",len(Y_train))
print("X_test len",len(X_test),"Y-test len",len(Y_test))


cleaned_emails=clean_text(X_train)
#print(cleaned_emails[3000])
term_docs=cv.fit_transform(cleaned_emails)
#term_docs_test=cv.transform(cleaned_test)
term_docs_train=cv.fit_transform(X_train)
label_index=get_label_index(Y_train)
term_docs_test=cv.transform(X_test)
clf=MultinomialNB(alpha=1.0,fit_prior=True)
clf.fit(term_docs_train, Y_train)
prediction_prob=clf.predict_proba(term_docs_test)
print(prediction_prob[0:10])
prediction=clf.predict(term_docs_test)
print(prediction[:10])
accuracy=clf.score(term_docs_test, Y_test)
print("the accuracy using MultinomialNB is: {0:.1f}%".format(accuracy*100))
print("confusion matrix=",confusion_matrix(Y_test,prediction,labels=[0,1]))
report=classification_report(Y_test,prediction)
print("\n",report)

pos_prob=prediction_prob[:,1]
thresholds=np.arange(0.0,1.2,0.1)
true_pos, false_pos=[0]*len(thresholds), [0]*len(thresholds)
for pred,y in zip(pos_prob,Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            # if truth and prediction are both 1
            if y==1:
                true_pos[i]+=1
            # if true is 0 and prediction is 1
            else:
                false_pos[i]+=1
        else:
            break


true_pos_rate=[tp/516.0 for tp in true_pos]
false_pos_rate=[fp/1191.0 for fp in false_pos]

plt.figure()
lw=2
plt.plot(false_pos_rate,true_pos_rate, color='darkorange', lw=lw)
plt.plot([0,1],[0,1], color='navy',lw=lw,linestyle="--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_test,pos_prob))
