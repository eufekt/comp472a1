# From my understanding, there seems to be 2 datasets (look at the dataset folder).
# This file will address the first one.

# one dataset is split into test, validation and training .csv files. I will merge the 3 files to create the distribution.

import pandas as pd;
import matplotlib.pyplot as plt
import csv
train_1 = 'dataset/train_1.csv'
# val_1 = 'dataset/val_1.csv'
# test_1 = 'dataset/test_1.csv'

df_train = pd.read_csv(train_1, usecols=[1024])
# df_val = pd.read_csv(val_1, usecols=[1024])
# df_test = pd.read_csv(test_1, usecols=[1024])

# frames = [df_train, df_val]
# df_result = pd.concat(frames)

df_train['1.828'].value_counts().plot.bar()

# df.plot.hist()
plt.show()

# print(df['1.828'].value_counts())


