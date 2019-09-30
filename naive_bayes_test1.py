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
# if x is a column of a dataset
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
##from sklearn.feature_extraction.text import CountVectorizer
##from sklearn.datasets import fetch_20newsgroups
##from nltk.corpus import names
###from nltk.corpus import shakespeare as sp
##
##import seaborn as sns
##import matplotlib.pyplot as plt
##import numpy as np    
##
##from nltk.stem import WordNetLemmatizer
##from sklearn.cluster import KMeans
##from sklearn.decomposition import NMF
##
##
##
##
##def letters_only(astr):
##    return(astr.isalpha())
##
##
###transformed=cv.fit_transform(groups.data)
####print(cv.get_feature_names())
##cv=CountVectorizer(stop_words="english",max_features=500)
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
##all_names=set(names.words())
###all_names=set(sp.words("macbeth.xml"))
###print(all_names)
##lemmatizer=WordNetLemmatizer()
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
