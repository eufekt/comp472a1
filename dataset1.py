# From my understanding, there seems to be 2 datasets (look at the dataset folder).
# This file will address the first one.

# one dataset is split into test, validation and training .csv files. I will merge the 3 files to create the distribution.

# import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.naive_bayes
import sklearn.metrics         # For confusion matrix, precision, recall, and f1-measure
import sklearn.svm             # For SVC class
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

#############
# Dataset 1 #
#############

#laods the csv files into a numpy array
train_1_csv = np.loadtxt(train_1, delimiter=',',  skiprows=0,) 
val_1_csv = np.loadtxt(value_1, delimiter=',',  skiprows=0)
letter_1 = np.loadtxt(info_1, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)

# test_no_label_1_csv = np.loadtxt(test_no_label_1, delimiter=',',  skiprows=0)
test_with_label_1_csv = np.loadtxt(test_with_label_1, delimiter=',',  skiprows=0, dtype='int32')

print(test_with_label_1_csv)



# split the matrices
train_1_Y = train_1_csv[:, -1] 
train_1_X = train_1_csv[:, :-1 ]


# fit the data
GNB = sklearn.naive_bayes.GaussianNB().fit(train_1_X, train_1_Y) # naive bayes GaussianNB



val_1_X = val_1_csv[:, :-1]
val_1_y = val_1_csv[:, -1]


val_1_y_predict = GNB.predict(val_1_X) # train the model

confusion_matrix_1 = sklearn.metrics.confusion_matrix(val_1_y, val_1_y_predict) # confusion matrix
report = sklearn.metrics.classification_report(val_1_y, val_1_y_predict, target_names=letter_1) # all the other stuff required


print(report)
# print(confusion_matrix_1)


test_with_label_1_X = test_with_label_1_csv[:, :-1 ]
test_with_label_1_Y = test_with_label_1_csv[:, -1]

test_with_label_1_Y_predict = GNB.predict(test_with_label_1_X)

confusion_matrix_1_test = sklearn.metrics.confusion_matrix(val_1_y, val_1_y_predict) # confusion matrix
report_1_test = sklearn.metrics.classification_report(test_with_label_1_X, test_with_label_1_Y_predict, target_names=letter_1) # all the other stuff required

print(report_1_test)



#############
# Dataset 2 #
#############

