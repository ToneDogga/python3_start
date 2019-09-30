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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np    

from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans




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

def letters_only(astr):
    return(astr.isalpha())


#transformed=cv.fit_transform(groups.data)
##print(cv.get_feature_names())
cv=CountVectorizer(stop_words="english",max_features=500)
groups=fetch_20newsgroups()

##sns.distplot(np.log(transformed.toarray().sum(axis=0)))
##plt.xlabel("log count")
##plt.ylabel("frequency")
##plt.title("distribution plot of 500 word sounts")
##plt.show()

cleaned=[]

all_names=set(names.words())
lemmatizer=WordNetLemmatizer()


for post in groups.data:
    cleaned.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))
transformed=cv.fit_transform(cleaned)
#print(cv.get_feature_names())
km=KMeans(n_clusters=20)
km.fit(transformed)
labels=groups.target
plt.scatter(labels,km.labels_)
plt.xlabel("Newsgroup")
plt.ylabel("Cluster")
plt.show()

