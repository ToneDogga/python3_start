import matplotlib.pyplot as plt
import numpy as np


def gini_impurity(labels):
    # when the set is empty, it is also pure
    if labels.size==0:
        return 0
    # count the occurances of each label
    counts=np.unique(labels,return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1-np.sum(fractions**2)

def entropy(labels):
    if labels.size==0:
        return 0
    counts=np.unique(labels,return_counts=True)[1]
    fractions = counts / float(len(labels))
    return -np.sum(fractions * np.log2(fractions))
    
def weighted_impurity(groups,criterion="gini"):
    # calculate weighted impurity of children after split
    total=sum(len(group) for group in groups)
    weighted_sum=0.0
    for group in groups:
        weighted_sum+=len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum    



def split_node(X,y,index,value):
    
    # split the data set X,y based on a feature and a value

    x_index=X[:, index]
    # if this feature is numerical
    if X[0,index].dtype.kind in ['i','f']:
        mask=x_index >= value
    else:
    # if this feature is categorical
        mask=x_index==value

    # split into left and right child
    left=[X[~mask,:],y[~mask]]
    right=[X[mask,:],y[mask]]
    return left,right


def get_best_split(X,y,criterion):
    # obtain the best splitting point and criterion
    # returns
    # dict{index: index of the feature, value : feature value, children: left and right children}

    best_index, best_value, best_score, children = None, None,1,None
    for index in range(len(X[0])):
        for value in np.sort(np.unique(X[:,index])):
            groups=split_node(X,y,index,value)
            impurity=weighted_impurity([groups[0][1],groups[1][1]], criterion)
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups

    return {"index":best_index,"value":best_value,"children":children}



def get_leaf(labels):
    # obtain a leaf as the majority of the labels
    return np.bincount(labels).argmax()


def split(node, max_depth, min_size, depth, criterion):
    # split the children of the node to construct new nodes or assign them to terminals
    # node is a dictionary with children info
    left,right=node["children"]
    del (node["children"])
    if left[1].size==0:
        node["right"]=get_leaf(right[1])
        return
    if right[1].size==0:
        node["left"]=get_leaf(left[1])
        return
    # check if current depth exceeds maximum depth
    if depth >= max_depth:
        node["left"],node["right"]=get_leaf(left[1]),get_leaf(right[1])
        return
    # check if left child has enough samples
    if left[1].size<=min_size:
        node["left"]=get_leaf(left[1])
    else:
        # if it has enough samples, we further split it
        result=get_best_split(left[0],left[1],criterion)
        result_left, result_right=result["children"]
        if result_left[1].size==0:
            node["left"]=get_leaf(result_right[1])
        elif result_right[1].size==0:
            node["left"]=get_leaf(result_left[1])
        else:
            node["left"]=result
            split(node["left"], max_depth, min_size, depth+1,criterion)
            
    # check if right child has enough samples
    if right[1].size<=min_size:
        node["right"]=get_leaf(right[1])
    else:
        # if it has enough samples, we further split it
        result=get_best_split(right[0],right[1],criterion)
        result_left, result_right=result["children"]
        if result_left[1].size==0:
            node["right"]=get_leaf(result_right[1])
        elif result_right[1].size==0:
            node["right"]=get_leaf(result_left[1])
        else:
            node["right"]=result
            split(node["right"], max_depth, min_size, depth+1,criterion)
 
def train_tree(X_train,y_train,max_depth,min_size, criterion="gini"):
    # construciton of the tree starts here
    X=np.array(X_train)
    y=np.array(y_train)
    root=get_best_split(X,y,criterion)
    split(root, max_depth, min_size, 1 , criterion)
    return root




    


##pos_fraction=np.linspace(0.00,1.00,1000)
##gini=1-pos_fraction**2-(1-pos_fraction)**2
##plt.plot(pos_fraction,gini)
##plt.ylim(0,1)
##plt.xlabel("positive fraction")
##plt.ylabel("Gini impurity")
##plt.show()
##
##
##print("{0:.4f}".format(gini_impurity([1,1,0,1,0])))
##
##print("{0:.4f}".format(gini_impurity([1,1,0,1,0,0])))
##print("{0:.4f}".format(gini_impurity([1,1,1,1])))
##
##
##children_1=[[1,0,1],[0,1]]
##children_2=[[1,1],[0,0,1]]
##
##print("Entropy of #1 split: {0:.4f}".format(weighted_impurity(children_1,"entropy")))
##                                            
##print("Entropy of #2 split: {0:.4f}".format(weighted_impurity(children_2,"entropy")))


def visualize_tree(node, depth=0):
    if isinstance(node,dict):
        if node["value"].dtype.kind in ['i','f']:
            condition=CONDITION["numerical"]
        else:
            condition=CONDITION["categorical"]

        print("{}|-X{} {} {}".format(depth * "  ", node["index"]+1, condition["no"], node["value"]))

        if "left" in node:
            visualize_tree(node["left"], depth+1)

        print("{}|-X{} {} {}".format(depth * "   ", node["index"]+1, condition["yes"], node["value"]))

        if "right" in node:
            visualize_tree(node["right"], depth+1)
    else:
        print("{}[{}]".format(depth * "  ", node))

                            

global CONDITION

CONDITION={"numerical":{"yes":">=","no":"<"},
           "categorical":{"yes":"is","no":"is not"}}

criterion_function={"gini":gini_impurity,"entropy":entropy}




X_train=[["tech","professional"],
         ["fashion","student"],
         ["fashion","professional"],
         ["sports","student"],
         ["tech","student"],
         ["tech","retired"],
         ["sports","professional"]]

y_train=[1,0,0,0,1,0,1]

tree=train_tree(X_train,y_train,2,2)

#print(X_train)
#print(y_train)


#print("{0:.4f}".format(gini_impurity(np.array(y_train))))


visualize_tree(tree)
           





X_train_n=[[6,7],
         [2,4],
         [7,2],
         [3,6],
         [4,7],
         [5,2],
         [1,6],
         [2,0],
         [6,3],
         [4,1]]

y_train_n=[0,0,0,0,0,1,1,1,1,1]



from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
tree_sk=DecisionTreeClassifier(criterion="gini",max_depth=2,min_samples_split=2)
tree_sk.fit(X_train_n,y_train_n)

#visualize_tree(tree_sk)

tree.export_graphviz(tree_sk, out_file="tree.dot",feature_names=["X1","X2"],impurity=False,filled=True,class_names=["0","1"])

