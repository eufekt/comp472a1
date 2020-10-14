# From my understanding, there seems to be 2 datasets (look at the dataset folder).
# This file will address the first one.

# one dataset is split into test, validation and training .csv files. I will merge the 3 files to create the distribution.

# import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.naive_bayes
import sklearn.preprocessing   # For scale function
import sklearn.metrics         # for accuracy_score
import sys

# prints entire entire instead of truncated
np.set_printoptions(threshold=sys.maxsize)


train_1 = 'dataset/train_1.csv'
value_1 = 'dataset/val_1.csv'
info_1 = 'dataset/info_1.csv'
test_no_label_1 = 'dataset/test_no_label_1.csv'
test_with_label_1 = 'dataset/test_with_label_1.csv'

train_2 = 'dataset/train_2.csv'
value_2 = 'dataset/val_2.csv'
info_2 = 'dataset/info_2.csv'
test_no_label_2 = 'dataset/test_no_label_2.csv'
test_with_label_2 = 'dataset/test_with_label_2.csv'



############################
## naive bayes GaussianNB ## 
############################

def naive_bayes_GaussianNB(train_csv, val_csv, test_with_label_csv, letter):
    # split the matrices
    train_Y = train_csv[:, -1] 
    train_X = train_csv[:, :-1 ]


    # fit the data
    GNB = sklearn.naive_bayes.GaussianNB().fit(train_X, train_Y) # training the naive bayes GaussianNB


    # split the array
    val_X = val_csv[:, :-1] # gets everything except last column
    val_y = val_csv[:, -1] # gets only last column

    # value prediction, I think this is useless
    val_y_predict = GNB.predict(val_X)


    test_with_label_X = test_with_label_csv[:, :-1 ]
    test_with_label_Y = test_with_label_csv[:, -1]

    # prediction 
    test_with_label_Y_predict = GNB.predict(test_with_label_X)

    # confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_with_label_Y, test_with_label_Y_predict) # confusion matrix

    # all the other stuff required
    report = sklearn.metrics.classification_report(test_with_label_Y, test_with_label_Y_predict, target_names=letter) 

    # classification array?
    index = np.arange(1, test_with_label_Y_predict.size + 1, 1) 
    array_to_write_to_file = np.stack((index, test_with_label_Y_predict), axis=1)

    return  array_to_write_to_file, report, confusion_matrix

#############
# Dataset 1 #
#############

#laods the csv files into a numpy array
train_1_csv = np.loadtxt(train_1, delimiter=',',  skiprows=0,) 
val_1_csv = np.loadtxt(value_1, delimiter=',',  skiprows=0)
letter_1 = np.loadtxt(info_1, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_1_csv = np.loadtxt(test_with_label_1, delimiter=',',  skiprows=0, dtype='int32')



array_to_write_to_file_1, report_1_test, confusion_matrix_1_test = naive_bayes_GaussianNB(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1)

print ("#############\n# dataset 1 #\n#############\n")
print("\nall\n")
print(array_to_write_to_file_1)
print("\n")
print("matrix dataset 1\n")
print(confusion_matrix_1_test)
print("\nreport\n")
print(report_1_test)




#############
# Dataset 2 #
#############



#laods the csv files into a numpy array
train_2_csv = np.loadtxt(train_2, delimiter=',',  skiprows=0,) 
val_2_csv = np.loadtxt(value_2, delimiter=',',  skiprows=0)
letter_2 = np.loadtxt(info_2, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_2_csv = np.loadtxt(test_with_label_2, delimiter=',',  skiprows=0, dtype='int32')


array_to_write_to_file_2, report_2_test, confusion_matrix_2_test = naive_bayes_GaussianNB(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2)

print ("#############\n# dataset 2 #\n#############\n")
print("\nall\n")
print(array_to_write_to_file_2)
print("\n")
print("matrix data set 2\n")
print(confusion_matrix_2_test)
print("\nreport\n")
print(report_2_test)