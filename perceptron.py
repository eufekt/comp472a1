
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.tree
import sklearn.preprocessing   # For scale function
import sklearn.metrics         # for accuracy_score
import sys

def per(train_csv, val_csv, test_with_label_csv, letter):
    # split the matrices
    train_Y = train_csv[:, -1] 
    train_X = train_csv[:, :-1]

    # fit the data
    perceptron = sklearn.linear_model.Perceptron().fit(train_X, train_Y) # training the Perceptron

    # split the array
    val_X = val_csv[:, :-1] # gets everything except last column
    val_y = val_csv[:, -1] # gets only last column

    # value prediction, I think this is useless
    val_y_predict = perceptron.predict(val_X)


    test_with_label_X = test_with_label_csv[:, :-1 ]
    test_with_label_Y = test_with_label_csv[:, -1]

    # prediction 
    test_with_label_Y_predict = perceptron.predict(test_with_label_X)

    # confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_with_label_Y, test_with_label_Y_predict) # confusion matrix

    # all the other stuff required
    report = sklearn.metrics.classification_report(test_with_label_Y, test_with_label_Y_predict, target_names=letter) 

    # classification array?
    index = np.arange(1, test_with_label_Y_predict.size + 1, 1) 
    array_to_write_to_file = np.stack((index, test_with_label_Y_predict), axis=1)

    return  {
        'arr': array_to_write_to_file,
        'report': report,
        'confusionMatrix':confusion_matrix
        }


