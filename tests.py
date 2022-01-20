import my_module
import pandas as pd

def test_invocation():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )

def test_return_types():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    
def test_features_dont_contain_target():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )
    assert target.name not in features.columns
    assert target.name == 'class'