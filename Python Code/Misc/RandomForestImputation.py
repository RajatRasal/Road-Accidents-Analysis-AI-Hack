import pandas as pd
import numpy as np

new_data = pd.read_csv("./new_data.csv")
data = pd.read_csv("./data.csv")

means_for_cols = data.mean()
cols_with_nulls = data.isnull().any(axis=0).index[:]

y = new_data.casualty_severity
X = new_data.drop('casualty_class', axis=1)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

continuous_variables = ['longitude', 'latitude']


def random_forest_generator(col_name, data):
    y = data[col_name]
    X = data.drop(col_name, axis=1)
    if col_name not in continuous_variables:
        rf = RandomForestClassifier(n_estimators=50, max_leaf_nodes=8, n_jobs=-1)
        rf.fit(X, y)
        return rf
    else:
        return


print("TRAINING")
col_to_rf = {col_name : random_forest_generator(col_name, new_data) for col_name in cols_with_nulls}

print("TRAINING DONE")

pickle.dump(col_to_rf, open("./rf_dict.pickle", "wb"))
print("DONE")
