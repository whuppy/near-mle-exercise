# Load libraries
import sys
import datetime
import numpy as np
import pandas as pd

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

# Load dataset
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = read_csv(url, names=names)
#print(dataset.head())
#ary = dataset.values
#print(ary)
#X = ary[:,0:4]
#print(X)
#y = ary[:,4]
#print(y)
#X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
#print(Y_train)
#print(Y_validation)

# Read in the training data:
# brand   category        geo     info_heuristic  actuals observed
# 25      10      825     1176.7644229082518      174     40
print(f'Reading training dataset {datetime.datetime.now().isoformat()} . . .')
train_data = read_csv('train.tsv',sep='\t',header=0)
print(train_data.head())
print(train_data.shape)
train_vals = train_data.values
#print(train_vals[:10])

# Split the training set into X (inputs) and y (result):
X = train_vals[:,np.r_[0:4,5]]
y = train_vals[:,5]
print(X)
print(y)
# Now split into training and validation sets:
print(f'Splitting training dataset into training and validation {datetime.datetime.now().isoformat()} . . .')
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
print(Y_train)
print(Y_validation)

sys.exit(1)

print('Building SVC model . . .')
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)



# Spot Check Algorithms
models = []
models.append(('SVM', SVC(gamma='auto')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    print(f'Evaluating model {name} {datetime.datetime.now().isoformat()} . . .')
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))




sys.exit(0)

# Read in the test data:
# brand   category        geo     info_heuristic  observed
# 94      9       656     1699.8687514725957      122
test_data = read_csv('test.tsv',sep='\t',header=0)
print(test_data.head())
print(test_data.shape)
test_vals = test_data.values

