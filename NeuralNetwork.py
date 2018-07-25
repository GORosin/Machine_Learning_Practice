import numpy as np
import pandas as pd

#for getting and processing data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data

#measure effeciency 
import time


#maybe implement relus later
def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.heaviside(x,0)

#using these for now
def sigmoid(x):
    return 1/(1+np.exp(np.negative(x)))

def sigmoid_derivative(x):
    sig=sigmoid(x)
    return sig*(1-sig)

class Network:
    def __init__(self,num_inputs=10,num_outputs=2,hidden_layers=[15,10]): #each element in array is the number of nodes in a hidden layer
        self.layers=[]                       #holds the weight and bias matrices of each layer
        self.num_outputs=num_outputs
        self.num_inputs=num_inputs
        first_layer_weights=np.random.normal(0,1,size=[hidden_layers[0],num_inputs])   #initialize wieghts with random numbers
        first_layer_bias=np.zeros((hidden_layers[0],1))                                #initialize bias with zeros
        self.layers.append([first_layer_weights,first_layer_bias])                  
        
        for i in range(1,len(hidden_layers)): #initialize hidden layers
            layer_weights=np.random.normal(0,1,size=[hidden_layers[i],hidden_layers[i-1]])
            layer_bias=np.zeros((hidden_layers[i],1))
            self.layers.append([layer_weights,layer_bias])

        last_layer_weights=np.random.normal(0,1,size=[num_outputs,hidden_layers[-1]])
        last_layer_bias=np.zeros((num_outputs,1))
        self.layers.append([last_layer_weights,last_layer_bias])
        self.neuron_layer=[]
        
    def forward_prop(self,input_layer):
        neurons=sigmoid(input_layer)
        self.neuron_layer=[] 
        for parameter in self.layers:
            self.neuron_layer.append(neurons) #save neurons for backprop
            neurons=sigmoid(np.matmul(parameter[0],neurons)+parameter[1])
            
        self.neuron_layer.append(neurons)
        return neurons

    def predict(self,input_layer):
        result=self.forward_prop(input_layer)
        return result
    
        
    def backward_prop(self,truth_data,learning_rate):
        delta=(self.neuron_layer[-1]-truth_data)*(self.neuron_layer[-1]*(1-self.neuron_layer[-1])) #delta is  the gradient (sort of)
        
        for layer_num in range(len(self.layers)-1,-1,-1): #goddamn arrays starting at zero
            weight_gradient=np.matmul(delta,np.transpose(self.neuron_layer[layer_num]))*learning_rate
            bias_gradient=delta*learning_rate
            
            delta=np.matmul(np.transpose(self.layers[layer_num][0]),delta)*(self.neuron_layer[layer_num]*(1-self.neuron_layer[layer_num]))

            self.layers[layer_num][0]-=weight_gradient
            self.layers[layer_num][1]-=bias_gradient
            
            
    def fit(self,features,labels):
        size=len(features)
        if size != len(labels):
            print("size of feaures must be size of labels")
            return
        for i in range(1,5000):
            great_filter=np.random.choice(a=[True, False], size=(size), p=[.95, .05])
            for data in range(size):
                if(great_filter[data]):
                    self.forward_prop(features[data])
                    self.backward_prop(labels[data],0.03)
                
def accuracy(truth,prediction):
    size=len(truth)
    correct=0
    for label in range(size):
        if(np.argmax(truth[label])==np.argmax(prediction[label])):
            correct+=1
    print(correct)
    print(size)



def TestWithIris():
    IrisNetwork=Network(4,3,[5,4])
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    array=dataset.as_matrix()
    target=['Iris-setosa','Iris-versicolor','Iris-virginica']
    #values = np.array(target)
    values=array[:,4]
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    
    labels= onehot_encoder.fit_transform(integer_encoded)
    data=array[:,0:4]
    data=np.array(data,dtype=float).reshape(150,4,1)
    labels=labels.reshape(150,3,1)

    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    print(labels.shape)
    print(data.shape)
    #IrisNetwork.fit(data[0:100],labels[0:100])


    
    pred=[]
    for i in data[100:150]:
        pred.append(IrisNetwork.predict(i))
    print(accuracy(labels[100:150],pred))

def TestWithMNist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #print(type(mnist))

    #train_images, train_labels = mndata.load_training()
    #test_dataset, test_labels = mndata.load_testing()
    train_dataset=mnist.train.images
    train_labels=mnist.train.labels
    valid_dataset=mnist.validation.images
    valid_labels=mnist.validation.labels
    
    test_dataset=mnist.test.images
    test_labels=mnist.test.labels
    
    test_dataset=test_dataset[0:100]
    test_labels=test_labels[0:100]
    
    train_dataset=train_dataset.reshape(55000,784,1)
    train_labels=train_labels.reshape(55000,10,1)

    test_dataset=test_dataset.reshape(100,784,1)
    test_labels=test_labels.reshape(100,10,1)
    print(train_dataset.shape)
    print(train_labels.shape)
    
    MNISTNetwork=Network(784,10,[370,180,90,40])

    
    rng_state = np.random.get_state()
    np.random.shuffle(train_dataset)
    np.random.set_state(rng_state)
    np.random.shuffle(train_labels)

    before=time.time()
    MNISTNetwork.fit(train_dataset,train_labels)
    after=time.time()
    print("time to fit: "+str(after-before))
    pred=[]
    for i in test_dataset[0:100]:
        pred.append(MNISTNetwork.predict(i))
        
    print(accuracy(test_labels[0:100],pred))

def TestWithWine():
    df = pd.read_csv('WineData.txt')
    array=df.as_matrix()
    values=array[:,0]
    data=array[:,1:]
    size=len(values)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    
    labels= onehot_encoder.fit_transform(integer_encoded)
    data=np.array(data,dtype=float).reshape(size,13,1)
    labels=labels.reshape(size,3,1)

    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    print(labels.shape)
    print(data.shape)
    WineNetwork=Network(13,3,[7,6])
    WineNetwork.fit(data[0:177],labels[0:177])


    
    pred=[]
    for i in data[0:177]:
        pred.append(WineNetwork.predict(i))
    print(accuracy(labels[0:177],pred))
    
#TestWithMNist()
#TestWithIris()
TestWithWine()
