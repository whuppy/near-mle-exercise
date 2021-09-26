# Load libraries
import numpy
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# brand   category        geo     info_heuristic  actuals observed
# 25      10      825     1176.7644229082518      174     40
train_data = read_csv('train.tsv',sep='\t',header=0)
print(train_data.head())
print(train_data.shape)
# Massage the training data so that 'actuals' is the last field.
train_vals = [t[numpy.r_[0:4,5,4]] for t in train_data.values]
print(train_vals[:10])

# brand   category        geo     info_heuristic  observed
# 94      9       656     1699.8687514725957      122
test_data = read_csv('test.tsv',sep='\t',header=0)
print(test_data.head())
print(test_data.shape)
test_vals = test_data.values

