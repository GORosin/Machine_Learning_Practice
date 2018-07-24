import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB #for comparison

#kernal functions
def Gaussian(x,mean,sigma):
    norm=1/(sigma*(2*math.pi)**0.5)
    return np.exp(-1*(x-mean)**2/(2*sigma**2))



class NaiveBayes:

    def __init__(self,num_inputs=10,num_outputs=3,targets=[]):
        self.data_size=np.zeros(num_outputs)
        self.probabilities=np.zeros(num_outputs)
        self.class_means=np.zeros((num_inputs,num_outputs))
        self.class_std=np.zeros((num_inputs,num_outputs))
        self.names={}
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        if targets:
            for name in range(len(targets)):
                self.names[targets[name]]=name
                
    def fit(self,data,classes):
        if(len(data) != len(classes)):
            print("error data must equal labels")
            return

        size=len(classes)
        labels=np.empty(size,dtype=int)
        if self.names:
            for i in range(len(labels)):
                labels[i]=int(self.names[classes[i]])
                
        unique, counts = np.unique(labels, return_counts=True)
        label_instances=dict(zip(unique, counts))
        
        for feature in range(self.num_inputs):
            for label in range(self.num_outputs):
                mask= labels==label
                data_mean=np.mean(data[mask,feature])
                data_variance=np.var(data[mask,feature])
                total_mean=self.data_size[label]*self.class_means[feature,label]+label_instances[label]*data_mean
                total_mean=total_mean/(self.data_size[label]+label_instances[label])

                total_std=self.data_size[label]*(self.class_std[feature,label]**2+self.class_means[feature,label]**2)
                total_std+=label_instances[label]*(data_variance+data_mean**2)
                total_std=total_std/(self.data_size[label]+label_instances[label]) - total_mean**2
    
                self.class_means[feature,label]=total_mean
                self.class_std[feature,label]=total_std**0.5
                
        for idx in range(self.num_outputs):
            self.data_size[idx]+=label_instances[idx]
            
    def predict(self,inputs):
        probabilities=np.empty(self.num_outputs)
        for i in range(self.num_outputs):
            x=Gaussian(inputs,self.class_means[:,i],self.class_std[:,i])
            probability=np.sum(np.log(x)) #log probability otherwise you're going to have a bad time
            probabilities[i]=probability

        targets = {y:x for x,y in self.names.items()}
        return targets[np.argmax(probabilities)]


def accuracy(truth,prediction):
    size=len(truth)
    correct=0
    for label in range(size):
        if(truth[label]==prediction[label]):
            correct+=1
    print(correct)
    print(size)

if __name__=="__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    array=dataset.as_matrix()
    target=['Iris-setosa','Iris-versicolor','Iris-virginica']
    NaiveIris=NaiveBayes(4,3,target)
    labels=array[:,4]
    data=array[:,0:4]
    data=np.array(data,dtype=float)
    
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    NaiveIris.fit(data[0:100],labels[0:100])
    data.reshape(150,4,1)
    predictions=[]
    for i in range(50):
        predictions.append(NaiveIris.predict(data[i+100]))

    print(accuracy(predictions,labels[100:150]))

