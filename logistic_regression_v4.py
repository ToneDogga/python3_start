import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


def sigmoid(input):
    return 1.0 / (1+np.exp(-input))

##x=np.linspace(-8,8,1000)
##y=sigmoid(x)
##
##plt.plot(x,y)
##plt.axhline(y=0,ls="dotted", color="k")
##plt.axhline(y=0.5,ls="dotted", color="k")
##plt.axhline(y=1,ls="dotted", color="k")
##plt.yticks([0.0, 0.25, 0.5, 0.75,1.0])
##plt.xlabel("x")
##plt.ylabel("y")
##plt.show()
##
##y_hat=np.linspace(0,1,1000)
##cost=-np.log(y_hat)
##plt.plot(y_hat,cost)
##plt.xlabel("prediction")
##plt.ylabel("cost")
##plt.xlim(0,1)
##plt.ylim(0,7)
##plt.show()
##
##y_hat=np.linspace(0,1,1000)
##cost=-np.log(1-y_hat)
##plt.plot(y_hat,cost)
##plt.xlabel("prediction")
##plt.ylabel("cost")
##plt.xlim(0,1)
##plt.ylim(0,7)
##plt.show()
##


def read_ad_click_data(n,offset=0):
    X_dict=[]
    y =[]     #np.array([])
    with open("train.csv","r") as csvfile:
        reader=csv.DictReader(csvfile)
        for i in range(offset):
            next(reader)
        i=0
        for row in reader:
            i+=1
            y.append(int(row['click']))
            del row['click'],row['id'],row['hour'],row['device_id'], row['device_ip']
            X_dict.append(row)
            if i >=n:
                break
    return X_dict, y
    



def compute_prediction(X,weights):
    # compute the prediction y_hat based on current weights
    # args
    # X (numpy.ndarray)
    # weights (numpy.ndarray)
    # returns
    # numpy.ndarray, y_hat of X under weights

    z=np.dot(X,weights)
    predictions=sigmoid(z)
    return predictions

def update_weights_sgd(X_train,y_train, weights, learning_rate):
    # update weights by one step
    # args
    # X_train, y_trains (numpy.ndarray, training data set)
    # weights (numpy.ndarray)
    # learning_rate (float)
    # returns
    # numpy.ndarray, updated weights
    #
    for X_each, y_each in zip(X_train, y_train):
        prediction=compute_prediction(X_each,weights)
        weights_delta=X_each.T*(y_each-prediction)
        weights+=learning_rate * weights_delta
    return weights


def update_weights_gd(X_train,y_train, weights, learning_rate):
    # update weights by one step
    # args
    # X_train, y_trains (numpy.ndarray, training data set)
    # weights (numpy.ndarray)
    # learning_rate (float)
    # returns
    # numpy.ndarray, updated weights
    #
    predictions=compute_prediction(X_train,weights)
    weights_delta=np.dot(X_train.T,y_train-predictions)
    m=y_train.shape[0]
    weights+=learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X,y,weights):
    # compute the cost of J(w)
    # args
    # X,y (numpy.ndarray, data set)
    # weights (numpy ndarray)
    # returns
    # float
    #
    predictions=compute_prediction(X,weights)
    cost=np.mean(-y*np.log(predictions)-(1-y)*np.log(1-predictions))
    return cost


def train_logistic_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    if fit_intercept:
        intercept=np.ones((X_train.shape[0],1))
        X_train=np.hstack((intercept,X_train))
    weights=np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weigths=update_weights_sgd(X_train,y_train,weights,learning_rate)
        if iteration%2==0:
            print(compute_cost(X_train,y_train,weights))
    return weights


def predict(X,weights):
    if X.shape[1]==weights.shape[0]-1:
        intercept=np.ones((X.shape[0],1))
        X=np.hstack((intercept,X))
    return compute_prediction(X,weights)

##
##X_train=np.array([[6,7],
##                  [2,4],
##                  [3,6],
##                  [4,7],
##                  [1,6],
##                  [5,2],
##                  [2,0],
##                  [6,3],
##                  [4,1],
##                  [7,2]])
##
##y_train=np.array([0,0,0,0,0,1,1,1,1,1])
##                
##
##
##weights=train_logistic_regression(X_train,y_train,max_iter=1000, learning_rate=0.1, fit_intercept=True)
##
##X_test=np.array([[6,1],
##                 [1,3],
##                 [3,1],
##                 [4,5]])
##
##predictions=predict(X_test,weights)
##print(predictions)
##
##plt.scatter(X_train[:,0],X_train[:,1],c=['b']*5+["k"]*5, marker='o')
##colours=["k" if prediction >=0.5 else 'b' for prediction in predictions]
##plt.scatter(X_test[:,0],X_test[:,1],marker="*",c=colours)
##plt.xlabel('x1')
##plt.ylabel('x2')
##plt.show()




##n=10000
##X_dict_train, y_train=read_ad_click_data(n)
dict_one_hot_encoder=DictVectorizer(sparse=False)
##X_train=dict_one_hot_encoder.fit_transform(X_dict_train)
##X_dict_test, y_test = read_ad_click_data(n,n)
##X_test=dict_one_hot_encoder.transform(X_dict_test)
##X_train_10k=X_train
##y_train_10k=np.array(y_train)
##
##import timeit
##start_time=timeit.default_timer()
##weights=train_logistic_regression(X_train_10k,y_train_10k, max_iter=25, learning_rate=0.01,fit_intercept=True)
##print("---- %0.3fs seconds ----------" %(timeit.default_timer()-start_time))
##
##X_test_10k=X_test
##predictions=predict(X_test_10k, weights)
##from sklearn.metrics import roc_auc_score
##print("the ROC AUC score on the testing set is: {0:.3f}".format(roc_auc_score(y_test,predictions)))

n=10000
X_dict_train, y_train=read_ad_click_data(n)
dict_one_hot_encoder=DictVectorizer(sparse=False)
X_train=dict_one_hot_encoder.fit_transform(X_dict_train)
##X_dict_test, y_test = read_ad_click_data(n,n)
##X_test=dict_one_hot_encoder.transform(X_dict_test)
X_train_10k=X_train
y_train_10k=np.array(y_train)
##
###import timeit
##start_time=timeit.default_timer()
##weights=train_logistic_regression(X_train_100k,y_train_100k, max_iter=5, learning_rate=0.01,fit_intercept=True)
##print("---- %0.3fs seconds ----------" %(timeit.default_timer()-start_time))
##
##X_test_10k=X_test
##predictions=predict(X_test_10k, weights)
###from sklearn.metrics import roc_auc_score
##print("the ROC AUC score on the testing ser is: {0:.3f}".format(roc_auc_score(y_test,predictions)))

X_dict_test, y_test_next10k=read_ad_click_data(10000,10000)
X_test_next10k=dict_one_hot_encoder.transform(X_dict_test)



print("sklearn no regulatization....\n\n")

from sklearn.linear_model import SGDClassifier
sgd_lr=SGDClassifier(loss="log", penalty=None, fit_intercept=True, max_iter=45, learning_rate="constant", eta0=0.01)

sgd_lr.fit(X_train_10k,y_train_10k)
predictions=sgd_lr.predict_proba(X_test_next10k)[:,1]
print("The ROC AUC on testing set is: {0:.3f}".format(roc_auc_score(y_test_next10k, predictions)))


print("sklearn with L1 regulization....\n\n")

l1_feature_selector=SGDClassifier(loss="log", penalty="l1", alpha=0.0001, fit_intercept=True, max_iter=5, learning_rate="constant", eta0=0.01)

l1_feature_selector.fit(X_train_10k,y_train_10k)
X_train_10k_selected=l1_feature_selector.partial_fit(X_train_10k)
print(X_train_10k_selected.shape)
print(X_train_10k.shape)
#predictions=sgd_lr.predict_proba(X_test_next10k)[:,1]
#print("The ROC AUC on testing set is: {0:.3f}".format(roc_auc_score(y_test_next10k, predictions)))


print(l1_feature_selector.coef_)



