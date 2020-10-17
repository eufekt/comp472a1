import numpy as np
import sys

from a_GNB import GNB
from b_Base_DT import base_DT
from c_Best_DT import best_DT
from d_PER import per
from e_Base_MLP import base_mlp
from f_Best_MLP import best_mlp

from utils import writeToFile

# prints entire entire instead of truncated
np.set_printoptions(threshold=sys.maxsize)

train_1 = '../dataset/train_1.csv'
value_1 = '../dataset/val_1.csv'
info_1 = '../dataset/info_1.csv'
test_no_label_1 = '../dataset/test_no_label_1.csv'
test_with_label_1 = '../dataset/test_with_label_1.csv'

train_2 = '../dataset/train_2.csv'
value_2 = '../dataset/val_2.csv'
info_2 = '../dataset/info_2.csv'
test_no_label_2 = '../dataset/test_no_label_2.csv'
test_with_label_2 = '../dataset/test_with_label_2.csv'


#############e
# Dataset 1 #
#############

#laods the csv files into a numpy array
train_1_csv = np.loadtxt(train_1, delimiter=',',  skiprows=0,) 
val_1_csv = np.loadtxt(value_1, delimiter=',',  skiprows=0)
letter_1 = np.loadtxt(info_1, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_1_csv = np.loadtxt(test_with_label_1, delimiter=',',  skiprows=0, dtype='int32')

# writeToFile('GNB_DS1', GNB(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1),)
# writeToFile('Base_DT_DS1', base_DT(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1)) # BASE
# writeToFile('Best_DT_DS1', best_DT(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1))
# writeToFile('PER_DS1', per(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1))
# writeToFile('BASE_MPL_DS1',base_mlp(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1)) # BASE
# writeToFile('BEST_MPL_DS1',best_mlp(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1))


#############
# Dataset 2 #
#############

#laods the csv files into a numpy array
train_2_csv = np.loadtxt(train_2, delimiter=',',  skiprows=0,) 
val_2_csv = np.loadtxt(value_2, delimiter=',',  skiprows=0)
letter_2 = np.loadtxt(info_2, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_2_csv = np.loadtxt(test_with_label_2, delimiter=',',  skiprows=0, dtype='int32')

# add all dataset 2 like with dataset 1


# For testing
GNB(train_2_csv, val_1_csv, test_with_label_2_csv, letter_2, prints=True)
base_DT(train_2_csv, val_1_csv, test_with_label_2_csv, letter_2, prints=True)
best_DT(train_2_csv, val_1_csv, test_with_label_2_csv, letter_2, prints=True)
per(train_2_csv, val_1_csv, test_with_label_2_csv, letter_2, prints=True)
base_mlp(train_2_csv, val_1_csv, test_with_label_2_csv, letter_2, prints=True)
best_mlp(train_2_csv, val_1_csv, test_with_label_2_csv, letter_2, prints=True)