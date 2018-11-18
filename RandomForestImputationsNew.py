import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

print("READING IN DATA")
new_data = pd.read_csv("./new_data.csv")
data = pd.read_csv("./data.csv")

print("CALCULATINF MEANS AND FINDING NULLS")
means_for_cols = data.mean()
cols_with_nulls = data.isnull().any(axis=0).index[:]

continuous_variables = ['longitude', 'latitude']

def random_forest_generator(col_name, data):
    y = data[col_name]
    X = data.drop(col_name, axis=1)
    if col_name not in continuous_variables:
        rf= RandomForestClassifier(max_depth=4,  n_estimators=50)
        rf.fit(X, y)
    else:
        rf = RandomForestRegressor(max_depth=4, n_estimators=50)
        rf.fit(X, y)
    return rf

print("DICT OF MODELS")
col_to_rf = {col_name : random_forest_generator(col_name, new_data)
             for col_name in cols_with_nulls}


pickle.dump(col_to_rf, open("./rf_dict.pickle", "wb"))
~                                                                                                             
