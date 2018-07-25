from DecisionTree import DecisionTree
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as ABoost
#boosting:
#run over the data, train a weak tree (depth=1 or 2)
#for each data point reweigh the data according to the following formula
#calculate total error
#error = sum(w(i) * terror(i)) / sum(w)
#where terror is 1 if classified incorrectly and 0 if correctly
#w(i) is the weight on each data point
#the tree is given the total_weight= ln((1-error) / error)
#each data point is given a weight w(i)= w(i) * exp(total_weight * terror)
#weights are initialized to 1/n
#the assumption is that data is binary (1 or -1)
#doesn't really work for multiclass data 
#use the data for survival rate for surgery for lung cancer (did/din't survive 1y)
#warning: this data is super depressing

class AdaBoost:
    def __init__(self,classifiers):
        self.number_of_classes=classifiers
        self.weights=[]
        self.classifiers=[]
        
    def create_classifier(self,inputs,outputs,weights):
        new_tree=DecisionTree(3,use_weights=True)
        new_tree.fit(inputs,outputs,weights)
        terror=np.empty(len(outputs),dtype=float)
        for indx,(data,truth) in enumerate(zip(inputs,outputs)):
            predict=new_tree.predict(data)
            terror[indx]=1.0-float(predict==truth)
        self.classifiers.append(new_tree)
        error=np.sum(weights*terror)/np.sum(weights)
        stage=np.log((1.0-error)/error)
        self.weights.append(stage)
        weights=weights*np.exp(stage*terror)
        return weights

    def fit(self,inputs,outputs):
        size=len(outputs)
        weights=np.full(size,1.0/float(size),dtype=float)
        for i in range(self.number_of_classes):
            weights=self.create_classifier(inputs,outputs,weights)

    def predict(self,outputs):
        outcome=0
        for i in range(self.number_of_classes):
            prediction=self.weights[i]*self.classifiers[i].predict(outputs)
            outcome+=prediction
        
        if outcome>0:
            return 1
        else:
            return -1

    def __str__(self):
        all_trees=''
        for i,tree in enumerate(self.classifiers):
            all_trees+="Tree: "+str(i)+"\n"
            all_trees+=tree.__str__()
            
        return all_trees
def process_cancer_data():
    #replacements
    changes={
        "T":1,"F":-1,
        "DGN3":3,"DGN2":2,"DGN4":4,"DGN6":6,"DGN5":5,"DGN8":7,"DGN1":1,
        "PRZ2":2,"PRZ1":1,"PRZ0":0,
        "OC11":1,"OC14":4,"OC12":2,"OC13":3
    }
    new_rows=[]
    with open('cancer_survival.txt', 'r') as f:
        reader = csv.reader(f) # pass the file to our csv reader
        for row in reader:     # iterate over the rows in the file
            new_row = row      # at first, just copy the row
            for key, value in changes.items(): # iterate over 'changes' dictionary
                new_row = [ x.replace(key, str(value)) for x in new_row ] # make the substitutions
            new_rows.append(new_row) # add the modified rows

    with open('cancer_data.csv', 'w') as f:
        # Overwrite the old file with the modified rows
        writer = csv.writer(f)
        writer.writerows(new_rows)
        
def accuracy(truth,prediction):
    size=len(truth)
    correct=0
    for label in range(size):
        if(truth[label]==prediction[label]):
            correct+=1
    print(correct)
    print(size)

if __name__=="__main__":
    BDT=AdaBoost(20)
    dataset = pd.read_csv('cancer_data.csv')
    array=dataset.as_matrix()
    labels=array[:,16]
    data=array[:,0:16]
    data=np.array(data,dtype=float)
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    BDT.fit(data,labels)
    print(BDT)
    predictions=[]
    for i in data:
        predictions.append(BDT.predict(i))
    accuracy(labels,predictions)

    tree_comparison=DecisionTree(3)
    tree_comparison.fit(data,labels)
    print("-----")
    predictions=[]
    for i in data:
        predictions.append(tree_comparison.predict(i))
    accuracy(labels,predictions)

    '''
    true_filter=labels>0
    false_filter=labels<0
    x=data[true_filter,4]
    y=data[false_filter,4]
    
    bins = np.linspace(0, 7, 100)
    plt.hist(x, bins, alpha=0.5, label='True')
    plt.hist(y, bins, alpha=0.5, label='False')
    plt.legend(loc='upper right')
    plt.show()

    
    Prof_BDT=ABoost()
    Prof_BDT.fit(data,labels)
    predictions=[]
    for i in data:
        predictions.append(Prof_BDT.predict(i.reshape(1,-1)))
    accuracy(labels,predictions)
    '''
