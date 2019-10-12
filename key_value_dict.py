
mydict = {1:["george",16,"test1",True],2:["amber",19,"test2",False]}

print(list(mydict.keys()))
print(list(mydict.values()))
print(list(mydict.items()))

print(mydict.keys())
print(mydict.values())
print(mydict.items())

# sammy.update({'online': False})


datastore = { "office": {
    "medical": [
      { "room-number": 100,
        "use": "reception",
        "sq-ft": 50,
        "price": 75
      },
      { "room-number": 101,
        "use": "waiting",
        "sq-ft": 250,
        "price": 75
      },
      { "room-number": 102,
        "use": "examination",
        "sq-ft": 125,
        "price": 150
      },
      { "room-number": 103,
        "use": "examination",
        "sq-ft": 125,
        "price": 150
      },
      { "room-number": 104,
        "use": "office",
        "sq-ft": 150,
        "price": 100
      }
    ],
    "parking": {
      "location": "premium",
      "style": "covered",
      "price": 750
    }
  }
}

print(datastore)
print(datastore["office"]["parking"])

print(datastore["office"]["medical"][1])

print(datastore["office"].get("law"))  
spaces = datastore['office']['medical']
print("spaces=",spaces)
# Here is a method to find and change a value in the database.
for item in spaces:
    if item.get('use') == "examination" :
       item['price'] = 175

for item in datastore['office']['medical']: # This loop shows the change is not only in books, but is also in database
    if item.get('use') == "examination" :
        print('The {} rooms now cost {}'.format(item.get("use"), item.get("price")))

people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'}}

print(people[1]['name'])
print(people[1]['age'])
print(people[1]['sex'])

#print(list(mydict.keys())[list(mydict.values()).index(19)]) # Prints george


#  Basically, it separates the dictionary's values in a list, finds the position of the value you have, and gets the key at that position.

#  More about keys() and .values() in Python 3: Python: simplest way to get list of values from dict?





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
###print(shakespeare_macbeth)
###print(sp["target_names"])
##
####print(groups.keys())
####print(groups["target_names"])
####print(groups.target)
####print(np.unique(groups.target))
####print(groups.data[0])
####print(groups.target[0])
####print(groups.target_names[groups.target[0]])
####print(len(groups.data[0]))
####print(len(groups.data[1]))
####sns.distplot(groups.target)
####plt.show()
##
##
##
##
##
##
####sns.distplot(np.log(transformed.toarray().sum(axis=0)))
####plt.xlabel("log count")
####plt.ylabel("frequency")
####plt.title("distribution plot of 500 word sounts")
####plt.show()
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
