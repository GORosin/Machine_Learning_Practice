import numpy as np

def shannon_entropy(arr):
    size=len(arr)    
    unique,counts=np.unique(arr,return_counts=True)
    prob=counts/size
    np.seterr(divide='ignore') #avoid the 0*log(0) problem 
    logs=prob*np.log2(prob)
    logs[np.isnan(logs)]=0
    total_entropy=-1*np.sum(logs)
    np.seterr(divide='warn') 
    return total_entropy


def mutual_information(inputs,outputs,cut):
    #mutual information is H(output)-p_<*H(output|x<cut)-p_>*H(output|x>cut)
    #where p_< is the fraction of the list less than the cut
    #p_> is the fraction of the list bigger than the cut
    #inputs should be a 1D array of *just one feature*
    
    H_output=shannon_entropy(outputs)
    size=len(outputs)
    output_filter=inputs>cut
    H_larger=(np.sum(output_filter)/size)*shannon_entropy(outputs[output_filter])
    H_smaller=(size-np.sum(output_filter))/size
    H_smaller=H_smaller*shannon_entropy(outputs[np.logical_not(output_filter)])

    delta_entropy=H_output-H_smaller-H_larger
    return delta_entropy


def feature_information(feature,outputs): #maximize mutual information in one feature
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
def max_information(inputs,outputs):
    max_feature_cut=0
    max_feature_information=0
    max_feature=0
    #find which feature gives us the best discrimination power
    for indx,feature in enumerate(inputs.T):
        cut,information=feature_information(feature,outputs)
        if(information>max_feature_information):
            max_feature_information=information
            max_feature_cut=cut
            max_feature=indx

    return (max_feature_cut,max_feature)

'''
Functions For Weighted Trees for Boosting
'''

#under the assumption that outputs are strictly 1 and -1
def weighted_entropy(outputs,weights):
    if(weights.shape[0]==0):
        return 0
    np.seterr(all='raise')
    label_one=outputs[outputs>0]
    label_two=outputs[outputs<0]

    freq_one=np.sum(weights[outputs>0])
    freq_two=np.sum(weights[outputs<0])
    if(freq_one==0 or freq_two==0):
        return 0
    size=freq_one+freq_two
    counts=np.array([freq_one,freq_two])
    np.seterr(divide='ignore')
    prob=counts/size
    logs=prob*np.log2(prob)
    logs[np.isnan(logs)]=0
    total_entropy=-1*np.sum(logs)
    np.seterr(divide='warn')
    
    return total_entropy

def weighted_information(inputs,outputs,cut,weights):
    #mutual information is H(output)-p_<*H(output|x<cut)-p_>*H(output|x>cut)
    #where p_< is the fraction of the list less than the cut
    #p_> is the fraction of the list bigger than the cut
    #inputs should be a 1D array of *just one feature*
    
    H_output=weighted_entropy(outputs,weights)
    size=np.sum(weights)
    output_filter=inputs>cut
    H_larger=(np.sum(weights[output_filter])/size)*weighted_entropy(outputs[output_filter],weights[output_filter])
    H_smaller=(size-np.sum(weights[output_filter]))/size
    H_smaller=H_smaller*weighted_entropy(outputs[np.logical_not(output_filter)],weights[np.logical_not(output_filter)])

    delta_entropy=H_output-H_smaller-H_larger
    return delta_entropy



        
def feature_weighted(feature,outputs,weights): #maximize mutual information in one feature
        max_entropy=0
        boundary=0
        #try to cut on each value in feature
        for cut in feature:
            delta_entropy=weighted_information(feature,outputs,cut,weights)
            if(delta_entropy>max_entropy):
                max_entropy=delta_entropy
                boundary=cut
                
        return(boundary,max_entropy)
                
#linear search through all of the data to  find the cut that maximizes mutual information
#i wonder if you can binary search here
def max_weighted(inputs,outputs,weights):
        max_feature_cut=0
        max_feature_information=0
        max_feature=0
        #find which feature gives us the best discrimination power
        for indx,feature in enumerate(inputs.T):
            cut,information=feature_weighted(feature,outputs,weights)
            if(information>max_feature_information):
                max_feature_information=information
                max_feature_cut=cut
                max_feature=indx

        return (max_feature_cut,max_feature)


#misc functions
def accuracy(truth,prediction):
    size=len(truth)
    correct=0
    for label in range(size):
        if(truth[label]==prediction[label]):
            correct+=1
    print(correct)
    print(size)

def is_pure(nparray):
    return np.all(nparray==nparray[0])
