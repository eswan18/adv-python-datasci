import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_features_and_target(csv_file, target_col):
    '''Split a CSV into a DF of numeric features and a target column.'''
    
    adult_census = pd.read_csv(csv_file)
    
    features = adult_census.drop(columns=target_col)
    target = adult_census[target_col]
    
    return (features, target)


def make_preprocessor(
    features,
    categorical_preprocessor=None,
    numeric_preprocessor=None
):
    '''
    Return a column transformer to prepare our features appropriately.
    
    Parameters
    ----------
    features:
        A dataframe of features
    categorical_preprocessor:
        A column transformer for categorical columns. Defaults to OneHotEncoder.
    '''
    if categorical_preprocessor is None:
        categorical_preprocessor= OneHotEncoder(handle_unknown="ignore")
    if numeric_preprocessor is None:
        numeric_preprocessor = StandardScaler()
    numeric_columns = features.select_dtypes(exclude=object).columns
    categorical_columns = features.select_dtypes(include=object).columns

    preprocessor = ColumnTransformer([
        ('categorical_encoder', categorical_preprocessor, categorical_columns),
        ('numeric_encoder', numeric_preprocessor, numeric_columns)
    ])
    return preprocessor