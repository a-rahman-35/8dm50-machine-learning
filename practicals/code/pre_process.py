import numpy as np

def min_max_scale(data):
    '''
        Scales data to [0,1] and returns this transformed data
        data:the dataset to be transformed
        '''
    X_max = np.max(data)
    X_min = np.min(data)
    return (data - X_min)/(X_max-X_min)

def normalize(data):
    '''
        Normalizes data and returns this transformed data
        data:the dataset to be transformed
        '''
    X_mean = np.mean(data)
    X_std = np.std(data)
    return (data - X_mean)/X_std