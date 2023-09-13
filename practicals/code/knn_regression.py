import scipy
import numpy as np

def knn_regression(k, X_train, y_train, X_test, y_test):
    '''
        Performs a k-nearest neighbor regression on the test dataset with a given training dataset
        Returns the predicted values on the test dataset
        k: k of the knn classifier
        X_train : the features of the train dataset
        y_train : the labels of the train dataset
        X_test : the features of the test dataset
        y_test : the labels of the test dataset
        mean_targets : mean of the closest targets as the prediction
        '''
    
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    
    # calculate pairwise differences using broadcasting
    pairwise_diff = X_train.reshape(num_samples, 1, num_features) - X_test
    
    # compute Euclidean distance
    euclidean_dist = np.sqrt((pairwise_diff**2).sum(2)).T
    
    # get indices of shortest Euclidean distances for all test samples to their neighbors
    sorted_indices = np.argsort(euclidean_dist)

    # select only the indices of the k closest points
    closest_indices = sorted_indices[:, :k]
    
    # find the corresponding target labels in y_train using these indices
    closest_targets = y_train[closest_indices]
    
    # compute mean of the closest targets
    mean_targets = np.mean(closest_targets, axis=1)

    return mean_targets


def knn_mse(y_test, y_pred):
    '''
        Evaluates performance of the algorithm using the mean squared error

        y_test : the labels of the test dataset
        y_pred : the predicted labels of the k-nn model
        mse: the mean squared error
    '''

    N = len(y_pred)
    mse = (1/N) * np.sum((y_pred-y_test)**2)
    
    return mse