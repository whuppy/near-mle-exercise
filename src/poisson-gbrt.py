import sys
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

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

td_train, td_test = train_test_split(train_data, test_size=0.20, random_state=66)

tree_preprocessor = ColumnTransformer(
            [
                ("categorical", OrdinalEncoder(), ["brand", "category", "geo"]),
                ("numeric", "passthrough", ["info_heuristic", "observed", ]),
            ],
            remainder="drop",
)

poisson_gbrt = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("regressor", HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=128)),
])
print(f'Fitting training data to Poisson GBRT {datetime.datetime.now().isoformat()} . . .')
poisson_gbrt.fit(td_train, td_train["actuals"], regressor__sample_weight=td_train["info_heuristic"])

print(f'Generating predictions from test data {datetime.datetime.now().isoformat()} . . .')
y_preds = poisson_gbrt.predict(td_test)
print(y_preds.shape, y_preds.min(), y_preds.mean(), y_preds.max())

mse = mean_squared_error(td_test["actuals"], y_preds, sample_weight=td_test["info_heuristic"]) 
print(f'MSE: {mse}')

mae = mean_absolute_error(td_test["actuals"], y_preds, sample_weight=td_test["info_heuristic"])
print(f'MAE: {mae}')

# Ignore non-positive predictions, as they are invalid for
# the Poisson deviance.
mask = y_preds > 0
if (~mask).any():
    n_masked, n_samples = (~mask).sum(), mask.shape[0]
    print(f"WARNING: Estimator yields invalid, non-positive predictions "
          f" for {n_masked} samples out of {n_samples}. These predictions "
          f"are ignored when computing the Poisson deviance.")
print("mean Poisson deviance: %.3f" %
      mean_poisson_deviance(td_test["actuals"][mask],
                            y_preds[mask],
                            sample_weight=td_test["info_heuristic"][mask]))

# Now that we have demonstrated that HistGradientBoostingRegressor works well,
# let's take it again from the top and train it on the entire train_data and
# then make predictions for the test_data.
poisson_gbrt = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("regressor", HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=128)),
])
print(f'Fitting entire train_data to Poisson GBRT {datetime.datetime.now().isoformat()} . . .')
poisson_gbrt.fit(train_data, train_data["actuals"], regressor__sample_weight=train_data["info_heuristic"])
print(f'Generating predictions from test data {datetime.datetime.now().isoformat()} . . .')
y_preds = poisson_gbrt.predict(test_data)
print(y_preds.shape, y_preds.min(), y_preds.mean(), y_preds.max())
print(f'Done {datetime.datetime.now().isoformat()}.')

