import numpy as np

def knn(k,train_X,train_y,test_X):
    '''
        Performs a binary knn-algorithm on the test dataset with a given training dataset
        Returns the classified predictions on the test dataset
        k: k of the knn classifier
        train_X : the features of the train dataset
        train_y : the labels of the train dataset
        test_X : the features of the test dataset
        y_pred : The classified predictions on the test dataset
    '''
    train_rows,_ = train_X.shape
    test_rows,_ = test_X.shape
    y_pred = []
    
    for i in range(test_rows):
        distances = []
        for j in range(train_rows):
            distances.append(np.sqrt(np.sum((train_X[j,:] - test_X[i,:])**2)))
            
        #select k neighbours with the closes distance to the tested element
        nearest_neighbours = np.argsort(distances)[:k]
        nearest_labels = [train_y[i] for i in nearest_neighbours]

        #count all the labels
        pos_total = nearest_labels.count(1)
        neg_total = nearest_labels.count(0)
        
        #assign label to test element with the highest vote count
        y_pred.append(int(pos_total > neg_total))


    return np.array(y_pred)

def knn_eval(y_true,y_pred, metric):
    '''
        Evaluates performance of the algorithm
        Returns evaluation score of algorithm with the chosen metric
        metric : The evaluation metric to use, "specificity", "sensitivity" or "accuracy"
        y_true : The labels of the test dataset
        y_pred : The classified predictions on the test dataset
    '''
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    if metric == "specificity":
        class_score = TN/(TN+FP)
    
    elif metric == "sensitivity":
        class_score = TP/(TP+FN)
    #accuracy standard metric
    else:
        class_score = (TP +TN)/(TP+TN+FN+FP)
    return class_score