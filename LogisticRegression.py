import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="white")

def logit_function(yhat):
    np.seterr(all='raise')
    return 1.0/(1.0+np.exp(np.negative(yhat)))

class LogisticRegression:
    def __init__(self,num_inputs):
        self.coeff=np.zeros(num_inputs+1)
        
    def get_prob(self,inputs):
        constant=self.coeff[0]
        yhat=np.sum(self.coeff[1:]*inputs)
        return logit_function(constant+yhat)

    def gradient_descent(self,inputs,outputs,learning_rate):
        for data,label in zip(inputs,outputs):
   
            yhat=self.get_prob(data)
            error=yhat-label
            gradient=learning_rate*(error* yhat* (1.0 - yhat))
            self.coeff[0]-= gradient
            self.coeff[1:]-= data*gradient

    def fit(self,data,labels):
        n_epochs=300
        for i in range(1,n_epochs):
            self.gradient_descent(data,labels,0.3)

    def predict(self,inputs):
        probability=self.get_prob(inputs)
        return int(probability>0.5)

    
    def plot(self,data,labels):
        xx, yy = np.mgrid[0:1:.01, 0:1:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = self.getprob(grid)[:, 1].reshape(xx.shape)
        plt.show()
        
def accuracy(truth,prediction):
    size=len(truth)
    correct=0
    for label in range(size):
        if(truth[label]==prediction[label]):
            correct+=1
    print(correct)
    print(size)
    return correct

if __name__=="__main__":
    dataset = pd.read_csv('diabetes.csv')
    array=dataset.as_matrix()
    labels=array[:,8]
    data=array[:,0:8]
    data=np.array(data,dtype=float)
    '''
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    '''
    #normalize data
    for row in range(len(data)):
        minimum=min(data[row])
        maximum=max(data[row])
        data[row]=(data[row]-minimum)/(maximum-minimum)

    x=np.linspace(0,2,150)
   
    
    clf=LogisticRegression(8)
    clf.fit(data,labels)
    
    predictions=[]
    for row in data:
        predictions.append(clf.predict(row))
        
    accuracy(labels,predictions)


