import numpy as np
import sys

from a_GNB import GNB
from b_Base_DT import base_DT
from c_Best_DT import best_DT
from d_PER import per
from e_Base_MLP import base_mlp
from f_Best_MLP import best_mlp
import pandas as pd;
import matplotlib.pyplot as plt

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

#demoTesting
# demo_train = '../dataset/train_2.csv'
# demo_value = '../dataset/val_2.csv'
# demo_info = '../dataset/info_2.csv'
# demo_test_no_label = '../dataset/test_no_label_2.csv'
# demo_test_with_label = '../dataset/test_with_label_2.csv'

# demo_train_csv = np.loadtxt(demo_train, delimiter=',',  skiprows=0,) 
# demo_val_csv = np.loadtxt(demo_value, delimiter=',',  skiprows=0)
# demo_letter = np.loadtxt(demo_info, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
# demo_test_with_label_csv = np.loadtxt(demo_test_with_label, delimiter=',',  skiprows=0, dtype='int32')

# writeToFile('DEMO_GNB_DS1', GNB(demo_train_csv, demo_val_csv, demo_test_with_label_csv, demo_letter, prints=True))


#############e
# Dataset 1 #
#############

#loads the csv files into a numpy array
train_1_csv = np.loadtxt(train_1, delimiter=',',  skiprows=0,) 
val_1_csv = np.loadtxt(value_1, delimiter=',',  skiprows=0)
letter_1 = np.loadtxt(info_1, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_1_csv = np.loadtxt(test_with_label_1, delimiter=',',  skiprows=0, dtype='int32')

# Distribution dataset 1

writeToFile('GNB_DS1', GNB(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1, prints=False))
writeToFile('BASE_DT_DS1', base_DT(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1, prints=False))
writeToFile('BEST_DT_DS1', best_DT(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1, prints=False))
writeToFile('PER_DS1', per(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1, prints=False))
writeToFile('BASE_MLP_DS1', base_mlp(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1, prints=False))
writeToFile('BEST_MLP_DS1', best_mlp(train_1_csv, val_1_csv, test_with_label_1_csv, letter_1, prints=False))

#############
# Dataset 2 #
#############

#loads the csv files into a numpy array
train_2_csv = np.loadtxt(train_2, delimiter=',',  skiprows=0,) 
val_2_csv = np.loadtxt(value_2, delimiter=',',  skiprows=0)
letter_2 = np.loadtxt(info_2, delimiter=',',  skiprows=1, usecols=1, dtype=np.str)
test_with_label_2_csv = np.loadtxt(test_with_label_2, delimiter=',',  skiprows=0, dtype='int32')

writeToFile('GNB_DS2', GNB(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2, prints=False))
writeToFile('BASE_DT_DS2',base_DT(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2, prints=False))
writeToFile('BEST_DT_DS2', best_DT(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2, prints=False))
writeToFile('PER_DS2', per(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2, prints=False))
writeToFile('BASE_MLP_DS2', base_mlp(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2, prints=False))
writeToFile('BEST_MLP_DS2', best_mlp(train_2_csv, val_2_csv, test_with_label_2_csv, letter_2, prints=False))


# Distribution dataset 1
data_1 = pd.read_csv(train_1, usecols=[1024])
plt.figure('DataSet 1')
data_1['1.828'].value_counts().plot.bar()

# Distribution dataset 2
data_2 = pd.read_csv(train_2, usecols=[1024])
plt.figure('DataSet 2')
data_2['9'].value_counts().plot.bar()
plt.show()