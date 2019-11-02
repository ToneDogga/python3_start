#import nltk
#nltk.download()
#
#from nltk.stem.porter import PorterStemmer
#porter_stemmer=PorterStemmer()
##
#print(porter_stemmer.stem("machines"))
#print(porter_stemmer.stem("learning"))
##
##
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
##
#print(lemmatizer.lemmatize("machines"))
#print(lemmatizer.lemmatize("learning"))
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np    

#from sklearn.feature_extraction.text import CountVectorizer

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





corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?']
#print("corpus=",corpus)


all_names=set(names.words())
lemmatizer=WordNetLemmatizer()
##


#cats = ['alt.atheism', 'sci.space']
cats = ['sci.space']

data_train = fetch_20newsgroups(subset='train', categories=cats, random_state=42)
##
#print(list(newsgroups_train.target_names))
label_train=data_train.target
data_train=clean_text(data_train.data)



vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()

#X = vectorizer.fit_transform(corpus)
X = vectorizer.fit_transform(data_train)

#print(vectorizer.get_feature_names())
#print("CV=",X.toarray())  
#input("?")

from sklearn.decomposition import NMF
nmf=NMF(n_components=20, random_state=43).fit(X)
for topic_idx, topic in enumerate(nmf.components_):
    label="{}:".format(topic_idx)
    print(label," ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))

##corpus = [
##     'This is the first document.',
##     'This document is the second document.',
##     'And this is the third one.',
##     'Is this the first document?']
##print("corpus=",corpus)
##vectorizer = TfidfVectorizer()
##Y = vectorizer.fit_transform(corpus)
##print(vectorizer.get_feature_names())
##print("TV=",np.around(Y.toarray(),2))  
##input("?")

       
def letters_only(astr):
    return(astr.isalpha())

##
##
###print(names.words()[:10])
###print(len(names.words()))
###input("?")
##
cats = ['alt.atheism']  #, 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
##
print(list(newsgroups_train.target_names))
##
##
##cv=CountVectorizer(stop_words="english",max_features=500)
##
##vectorizer = TfidfVectorizer()
###vectorizer = CountVectorizer()
##
##vectors = vectorizer.fit_transform(newsgroups_train.data)
##print("vectors.shape",vectors.shape)
##print(vectorizer.get_feature_names())
##input("?")
##mylist=np.around(vectors[0].toarray(),2).tolist()
##print(mylist)
###print(', '.join(mylist))
##
####groups=fetch_20newsgroups()
####print("groups.keys()=\n",groups.keys())
####input("?")
####print("groups.target_names=\n",groups["target_names"])
####input("?")
#####print("groups.description=\n",groups["description"])
#####input("?")
####print("groups.target=\n",groups["target"])
####input("?")
####print("groups.filenames=\n",groups["filenames"])
####input("?")
####print("groups.DESCR=\n",groups["DESCR"])
####input("?")
##
##print("groups.data=\n",groups.data[0])
##input("?")
##
##print("train filenames shape",newsgroups_train.filenames.shape)
##input("?")
##print("train target shape",newsgroups_train.target.shape)
##input("?")  
##print("newsgroups train target",newsgroups_train.target[0])
##input("?")
##transformed=cv.fit_transform(groups.data)
##print("feature names=\n",cv.get_feature_names())
##input("?")
##
####
##cleaned=[]
####
##all_names=set(names.words())
##lemmatizer=WordNetLemmatizer()
##
####
##for post in groups.data:
##    cleaned.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))
##transformed=cv.fit_transform(cleaned)
##print("transformed=\n",transformed)
##input("?")
###print(cv.get_feature_names())
###km=KMeans(n_clusters=20)
###km.fit(transformed)
##labels=groups.target
##print("labels=\n",labels)
###plt.scatter(labels,km.labels_)
###plt.xlabel("Newsgroup")
###plt.ylabel("Cluster")
###plt.show()

