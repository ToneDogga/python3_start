from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.corpus import shakespeare as sp

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np    

from nltk.stem import WordNetLemmatizer
##from sklearn.cluster import KMeans
from sklearn.decomposition import NMF




def letters_only(astr):
    return(astr.isalpha())


#transformed=cv.fit_transform(groups.data)
##print(cv.get_feature_names())
cv=CountVectorizer(stop_words="english",max_features=500)
#groups=fetch_20newsgroups()


print(sp.fileids())
macbeth = sp.words("macbeth.xml")
print(macbeth)
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

cleaned=[]

all_names=set(names.words())
macbeth=set(sp.words("macbeth.xml"))
#print(all_names)
lemmatizer=WordNetLemmatizer()


for post in macbeth:   #all_names:  #groups.data:
    cleaned.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))
 #   if letters_only(post):
  #      cleaned.append(" ".join([lemmatizer.lemmatize(post.lower())]))   # if letters_only(post)]))

#print(cleaned)
transformed=cv.fit_transform(cleaned)
#print(cv.get_feature_names())
#km=KMeans(n_clusters=20)
#km.fit(transformed)
nmf=NMF(n_components=100,random_state=43).fit(transformed)
print("nmf=\n",nmf.components_.tolist())
for topic_idx, topic in enumerate(nmf.components_):
    label="{}: ".format(topic_idx)
    print(label," ".join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))
    
##labels=groups.target
##plt.scatter(labels,km.labels_)
##plt.xlabel("Newsgroup")
##plt.ylabel("Cluster")
##plt.show()

