# SGDC Stochastic gradient descent classifier
# written by Anthony Paech  29/11/19
#
# Take a excel spreadsheet that is a slice of the salestrans
# ideally, filtered by customer code and product group and saved as a CSV
# such that the features are simply
# "productcode" "day_delta"
# and the classes are binned values of the "qty"
# these are read into a dictionary (the DictVectoriser is forgiving and very good)
# and vectorised with one hot encoder
#  also test the accuracy with other models like naive bayes, random forest and Support vector machine
#
# pseudocode
# load slice of sales trans straight from excel
# choose columns X,y
# cleanup NaN's
# convert date to day_delta
# create order_day_delta (the day gap between the last order and this one)
# split into X_train, X_test, y_train, y_test
# convert X_train and X_test to dictionaries
# convert y_train, y_test to np.array()
# bin the Y_train, Y_test data
# use DictVectorizer on X_train
# use SGDClassifier object
# run using GridSearch
# use ROC AUC to score predictions on y_test based on X_test
#




import numpy as np
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

def day_delta(df):

#    df['day_delta'] = (df.date.max()-df.date).dt.days.astype(int)
    df['day_delta'] = (df.date-df.date.min()).dt.days
    df.drop(columns='date',axis=1,inplace=True)
 #   df.drop(df['date'],inplace=True)
##    date_N_days_ago = datetime.now() - timedelta(days=2)
##
##    print(datetime.now())
##    print(date_N_days_ago)
    return df

def read_in_csvdata(filename,n, offset=0):
    X_dict, y = [], []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i in range(offset):
            next(reader)
        i = 0
        for row in reader:
            i += 1
            y.append(int(row['qty']))
            del row['qty'] #, row['location'], row['code'], row['refer'], row['linenumber']
            X_dict.append(row)
            if i >= n:
                break
    return X_dict, y


def read_in_exceldata(filename,n):   #,offset=0):
    df=pd.read_excel(filename, "Sheet1")
    X=df.iloc[0:n,:2]
    Y=df.iloc[0:n,4].values
#    print(X)
    X=day_delta(X)
  #  print("X=\n",X)
    #print(Counter(Y))
  #  X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)
  #  print("x train=",X_train[:10])
  #  X_dict_train=dict(X_train)
    #d = dict(enumerate(myarray.flatten(), 1))
  #  X_dict_test=dict(X_test)
    return X_dict_train,X_dict_test,y_train,y_test


def read_excel(filename,rows):
    xls = pd.ExcelFile(filename)    #'salestransslice1.xlsx')
    if rows==-1:
        return xls.parse(xls.sheet_names[0])
    else:        
        return xls.parse(xls.sheet_names[0]).head(rows)


def write_excel(df,outfilename):
    df.to_excel(outfilename)    #'salestransslice1.xlsx')
    return
 
def write_csv(df,outfilename):
    df.to_csv(outfilename,index=False)    #'salestransslice1.xlsx')
    return
 


def discrete_ybins(array,bins):
    # divide the classes (or features) into a bucket number of equally sized ranges
    return pd.cut(array.astype(float),bins,labels=range(len(bins)-1),right=True,retbins=False)  #.astype(int)
     

n = 3000
n2=800
bins=[0,8,16,10000]
infile="salestransslice4-FLNOR.xlsx"
outfile="salestransslice4-FLNOR.csv"
df=read_excel(infile,-1)
df=day_delta(df)

df['qty_bins']=pd.cut(df['qty'].values.astype(float),bins,labels=range(len(bins)-1),right=True,retbins=False)
print("df=\n",df.to_string())
df.dropna(inplace=True)
write_csv(df,outfile)



X_dict,y_list = read_in_csvdata(outfile,n,offset=0)

#print("X_dict=\n",X_dict[:10],"y_list=\n",y_list[:10])
#print(list(X_dict)[:10])
#X_train, X_test, y_train, y_test = train_test_split(list(X_dict),y_list, test_size=0.2,random_state=42)


dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict)
#print("X_train=\n",X_train)


y_train = discrete_ybins(np.array(y_list),bins)
#print("y_train=\n",y_train)



X_dict,y_list = read_in_csvdata("salestransslice4-FLNOR.csv",n2,offset=n)
X_test = dict_one_hot_encoder.transform(X_dict)
print("X_test=\n",X_test[:10])


y_test= discrete_ybins(np.array(y_list),bins)
print("y_test=\n",y_test[:10])

# Use scikit-learn package

sgd_lr = SGDClassifier(loss='log', penalty=None, fit_intercept=True, max_iter=500, learning_rate='constant', eta0=0.01)
sgd_lr.fit(X_train, y_train)

predictions = sgd_lr.predict_proba(X_test)[:, 1]
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test, predictions)))



# Feature selection with L1 regularization
##
##l1_feature_selector = SGDClassifier(loss='log', penalty='l1', alpha=0.0001, fit_intercept=True, max_iter=5, learning_rate='constant', eta0=0.01)
##l1_feature_selector.fit(X_train_10k, y_train_10k)
###X_train_10k_selected = l1_feature_selector.transform(X_train_10k)
##print(X_train_10k_selected.shape)
##print(X_train_10k.shape)
##
### bottom 10 weights and the corresponding 10 least important features
##print(np.sort(l1_feature_selector.coef_)[0][:10])
##print(np.argsort(l1_feature_selector.coef_)[0][:10])
### top 10 weights and the corresponding 10 most important features
##print(np.sort(l1_feature_selector.coef_)[0][-10:])
##print(np.argsort(l1_feature_selector.coef_)[0][-10:])





# Multiclass classification with logistic regression

##from sklearn.feature_extraction.text import TfidfVectorizer
##from sklearn.datasets import fetch_20newsgroups
##from sklearn.linear_model import SGDClassifier
##from nltk.corpus import names
##from nltk.stem import WordNetLemmatizer

##all_names = set(names.words())
##lemmatizer = WordNetLemmatizer()
##
##def letters_only(astr):
##    for c in astr:
##        if not c.isalpha():
##            return False
##    return True
##
##def clean_text(docs):
##    cleaned_docs = []
##    for doc in docs:
##        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
##                                        for word in doc.split()
##                                        if letters_only(word)
##                                        and word not in all_names]))
##    return cleaned_docs
##
##data_train = fetch_20newsgroups(subset='train', categories=None, random_state=42)
##data_test = fetch_20newsgroups(subset='test', categories=None, random_state=42)
##
##cleaned_train = clean_text(data_train.data)
##label_train = data_train.target
##cleaned_test = clean_text(data_test.data)
##label_test = data_test.target
##
##tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', max_features=40000)
##term_docs_train = tfidf_vectorizer.fit_transform(cleaned_train)
##term_docs_test = tfidf_vectorizer.transform(cleaned_test)

# combined with grid search
from sklearn.model_selection import GridSearchCV
parameters = {'penalty': ['l2', None],
              'alpha': [1e-07, 1e-06, 1e-05, 1e-04],
              'eta0': [0.01, 0.1, 1, 10]}

sgd_lr = SGDClassifier(loss='log', learning_rate='constant', eta0=0.01, fit_intercept=True, max_iter=50)

grid_search = GridSearchCV(sgd_lr, parameters, n_jobs=-1, cv=3)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

sgd_lr_best = grid_search.best_estimator_
accuracy = sgd_lr_best.score(X_test, y_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
