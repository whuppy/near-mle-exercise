import sys
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

print(f'Reading training dataset {datetime.datetime.now().isoformat()} . . .')
train_data = pd.read_csv('/home/ec2-user/near-mle-exercise/rundir/train.tsv',sep='\t',header=0)
print(train_data.head())
print(train_data.shape)

print(f'Reading test dataset {datetime.datetime.now().isoformat()} . . .')
test_data = pd.read_csv('/home/ec2-user/near-mle-exercise/rundir/test.tsv',sep='\t',header=0)
print(test_data.head())
print(test_data.shape)

td_train, td_test = train_test_split(train_data, test_size=0.20, random_state=1)

tree_preprocessor = ColumnTransformer(
            [
                ("categorical", OrdinalEncoder(), ["brand", "category", "geo"]),
                ("numeric", "passthrough", ["info_heuristic", "observed", "actuals"]),
            ],
            remainder="drop",
)

poisson_gbrt = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("regressor", HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=128)),
])
poisson_gbrt.fit(td_train, td_train["actuals"], regressor__sample_weight=td_train["info_heuristic"])
preds = poisson_gbrt.predict(td_test)
print(preds.shape, preds.min(), preds.mean(), preds.max())
