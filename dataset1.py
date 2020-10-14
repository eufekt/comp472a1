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

#############
# Dataset 1 #
#############

#laods the csv files into a numpy array
train_1_csv = np.loadtxt(train_1, delimiter=',',  skiprows=0,) 
val_1_csv = np.loadtxt(value_1, delimiter=',',  skiprows=0)
letter_1 = np.loadtxt(info_1, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_1_csv = np.loadtxt(test_with_label_1, delimiter=',',  skiprows=0, dtype='int32')


# split the matrices
train_1_Y = train_1_csv[:, -1] 
train_1_X = train_1_csv[:, :-1 ]


# fit the data
GNB = sklearn.naive_bayes.GaussianNB().fit(train_1_X, train_1_Y) # naive bayes GaussianNB


# split the array
val_1_X = val_1_csv[:, :-1] # gets everything except last column
val_1_y = val_1_csv[:, -1] # gets only last column


val_1_y_predict = GNB.predict(val_1_X) # train the model


test_with_label_1_X = test_with_label_1_csv[:, :-1 ]
test_with_label_1_Y = test_with_label_1_csv[:, -1]

test_with_label_1_Y_predict = GNB.predict(test_with_label_1_X)

confusion_matrix_1_test = sklearn.metrics.confusion_matrix(val_1_y, val_1_y_predict) # confusion matrix
report_1_test = sklearn.metrics.classification_report(test_with_label_1_Y, test_with_label_1_Y_predict, target_names=letter_1) # all the other stuff required

index = np.arange(1, test_with_label_1_Y_predict.size + 1, 1) 
array_to_write_to_file = np.stack((index, test_with_label_1_Y_predict), axis=1)

print("\nall\n")
print(array_to_write_to_file)
print("\n")
print("matrix\n")
print(confusion_matrix_1_test)
print("\nreport\n")
print(report_1_test)



#############
# Dataset 2 #
#############

