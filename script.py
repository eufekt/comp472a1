# From my understanding, there seems to be 2 datasets (look at the dataset folder).
# This file will address the first one.

# one dataset is split into test, validation and training .csv files. I will merge the 3 files to create the distribution.

# import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.naive_bayes
import sklearn.tree
import sklearn.preprocessing   # For scale function
import sklearn.metrics         # for accuracy_score
import sys
import gnb
import base_dt

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


#############e
# Dataset 1 #
#############

#laods the csv files into a numpy array
train_1_csv = np.loadtxt(train_1, delimiter=',',  skiprows=0,) 
val_1_csv = np.loadtxt(value_1, delimiter=',',  skiprows=0)
letter_1 = np.loadtxt(info_1, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_1_csv = np.loadtxt(test_with_label_1, delimiter=',',  skiprows=0, dtype='int32')

array_GNB_1, report_GNB_1_test, confusion_matrix_GNB_1_test = gnb.naive_bayes_GaussianNB(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1)

array_BASE_DT_1, report_BASE_DT_1_test, confusion_matrix_BASE_DT_1_test = base_dt.base_DT(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1)



#############
# Dataset 2 #
#############

#laods the csv files into a numpy array
train_2_csv = np.loadtxt(train_2, delimiter=',',  skiprows=0,) 
val_2_csv = np.loadtxt(value_2, delimiter=',',  skiprows=0)
letter_2 = np.loadtxt(info_2, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_2_csv = np.loadtxt(test_with_label_2, delimiter=',',  skiprows=0, dtype='int32')

array_to_write_to_file_GNB_2, report_GNB_2_test, confusion_matrix_GNB_2_test = gnb.naive_bayes_GaussianNB(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2)

array_BASE_DT_2, report_BASE_DT_2_test, confusion_matrix_BASE_DT_2_test = base_dt.base_DT(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2)



# utils
# print array to File
# f = open("outputs/GNB-DS1", "w")
# for item in array_GNB_1:
#     f.write(str(item[0]).split('.')[0]+","+str(item[1]).split('.')[0]+"\n")
# f.close()