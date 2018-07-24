import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as SKDT
import time
from TreeUtils import *

class node:
    def __init__(self,feature,cut,label=None,left=None,right=None):
        self.feature=feature #feature to cut on
        self.cut=cut #cut on above feature
        self.left=left
        self.right=right
        self.label=label #for leaf nodes, the class we ultimatley get

    def is_leaf(self):
        return (self.left==None and  self.right==None)
    
class DecisionTree:
    
    def __init__(self,depth,use_weights=False,inputs=4,outputs=3):
        self.depth=depth
        self.root=None
        self.use_weights=use_weights
        
    def feature_information(self,feature,outputs): #maximize mutual information in one feature
        max_entropy=0
        boundary=0
        #try to cut on each value in feature
        for cut in feature:
            delta_entropy=mutual_information(feature,outputs,cut)
            if(delta_entropy>max_entropy):
                max_entropy=delta_entropy
                boundary=cut
                
        return(boundary,max_entropy)
                
    #linear search through all of the data to  find the cut that maximizes mutual information
    #i wonder if you can binary search here
    def max_information(self,inputs,outputs):
        max_feature_cut=0
        max_feature_information=0
        max_feature=0
        #find which feature gives us the best discrimination power
        for indx,feature in enumerate(inputs.T):
            cut,information=self.feature_information(feature,outputs)
            if(information>max_feature_information):
                max_feature_information=information
                max_feature_cut=cut
                max_feature=indx

        return (max_feature_cut,max_feature)

    def create_tree(self,inputs,outputs,current_depth):
        if(current_depth>=self.depth or  is_pure(outputs)):
            unique,counts=np.unique(outputs,return_counts=True)
            try:
                best_label_guess=unique[np.argmax(counts)]
                #best guess for label is  which label appears most  often in what's left of our output space
            except ValueError:
                best_label_guess=None
            return node(None,None,best_label_guess)
        
        else:
            cut,feature=self.max_information(inputs,outputs)
            left_mask=inputs.T[feature]>cut #cut on all the data bigger than threshold
            right_mask=np.logical_not(left_mask) #cut on data less than threshold
            return node(feature,cut,label=None,
                        left=self.create_tree(inputs[left_mask],outputs[left_mask],current_depth+1),
                        right=self.create_tree(inputs[right_mask],outputs[right_mask],current_depth+1))


    def create_weighted_tree(self,inputs,outputs,current_depth,weights):
        if(current_depth>=self.depth or is_pure(outputs)):
            freq_one=np.sum(weights[outputs>0])
            freq_two=np.sum(weights[outputs<0])
            if(freq_one==freq_two):
                return np.random.choice([-1,1])
            try:
                best_label_guess=(freq_one-freq_two)/abs(freq_one-freq_two)
                #best guess for label is  which label appears most  often in what's left of our output space
            except ValueError:
                best_label_guess=None
            return node(None,None,best_label_guess)
        
        else:
            cut,feature=max_weighted(inputs,outputs,weights)
            left_mask=inputs.T[feature]>cut #cut on all the data bigger than threshold
            right_mask=np.logical_not(left_mask) #cut on data less than threshold
            return node(feature,cut,label=None,
                        left=self.create_weighted_tree(inputs[left_mask],outputs[left_mask],current_depth+1,weights[left_mask]),
                        right=self.create_weighted_tree(inputs[right_mask],outputs[right_mask],current_depth+1,weights[right_mask]))
        
    def fit(self,inputs,outputs,weights=[]):
        if(self.use_weights):
            self.root=self.create_weighted_tree(inputs,outputs,0,weights)
        else:
            self.root=self.create_tree(inputs,outputs,0)
        

    def predict(self,inputs):
        pointer=self.root
        while(not pointer.is_leaf()):
            feature=pointer.feature
            if inputs.T[feature]>pointer.cut:
                pointer=pointer.left
            else:
                pointer=pointer.right

        return pointer.label

    def to_str(self,node,level):
        if(node.is_leaf()):
            node_str="\t"*level+"----------\n"            
            node_str+= "\t"*level+str(node.label)+"\n"
            node_str+="\t"*level+"----------\n"
            return node_str
        else:
            node_str="\t"*level+"----------\n"
            node_str+="\t"*level+"feature="+ str(node.feature)+"\n"+"\t"*level+"cut= "+str(node.cut)+"\n"
            node_str+="\t"*level+"----------\n"
            return node_str+self.to_str(node.left,level+1)+self.to_str(node.right,level+1)

    def __str__(self):
        return self.to_str(self.root,0)
    
if __name__=="__main__":
    depth=2
    DT=DecisionTree(depth=depth)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    array=dataset.as_matrix()
    target=['Iris-setosa','Iris-versicolor','Iris-virginica']
    labels=array[:,4]
    data=array[:,0:4]
    data=np.array(data,dtype=float)
    
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    #DT.fit(data[0:100],labels[0:100])
    x_ex=np.array([[1,4,3,4],[4,8,6,8],[7,4,9,10],[11,8,13,14]])
    y_ex=np.array([0,1,0,1])
    my_begin=time.time()

    DT.fit(data[0:100],labels[0:100])
    predictions=[]
    my_end=time.time()

    for i in range(50):
        predictions.append(DT.predict(data[i+100]))

    print("time for my tree to fit")
    print(my_end-my_begin)
    
    professional_predictions=[]
    professional_tree=SKDT(criterion='entropy',max_depth=depth)
    sk_begin=time.time()

    professional_tree.fit(data[0:100],labels[0:100])
    sk_end=time.time()

    for i in range(50):
        professional_predictions.append(professional_tree.predict(data[i+100].reshape(1,-1)))

    print("time for sklearn to fit")
    print(sk_end-sk_begin)
    print("my tree accuracy")
    accuracy(predictions,labels[100:150])
    print("sklearn tree accuracy")    
    accuracy(professional_predictions,labels[100:150])
    print(DT)
