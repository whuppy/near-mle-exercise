# Load libraries
import sys
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Read in the training data:
# brand   category        geo     info_heuristic  actuals observed
# 25      10      825     1176.7644229082518      174     40
print(f'Reading training dataset {datetime.datetime.now().isoformat()} . . .')
train_data = pd.read_csv('train.tsv',sep='\t',header=0)
print(train_data.head())
print(train_data.shape)
# Massage the training data so that 'actuals' is the last field.
train_vals = numpy.array([t[numpy.r_[0:4,5,4]] for t in train_data.values])
#print(train_vals[:10])

# Split the training set into X (inputs) and y (result):
X = train_vals[:,0:5]
y = train_vals[:,5]
print(X)
print(y)
