import pandas as pd
import numpy as np
import pytest
from src.data_preprocessor import DataPreprocessor

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'num1': [1, 2, np.nan, 4],
        'num2': [10, 20, 30, np.nan],
        'cat1': ['A', 'B', np.nan, 'A'],
        'cat2': ['X', np.nan, 'X', 'Y'],
        'good_num': [1, 2, 3, 4],
        'high_nan': [np.nan, np.nan, np.nan, 1]  
    })

def test_init_invalid():
    with pytest.raises(ValueError):
        DataPreprocessor('not_df')

def test_remove_missing(sample_df):
    prep = DataPreprocessor(sample_df)
    df1 = prep.remove_missing(0.5)
    assert 'high_nan' not in df1.columns
    assert not df1.isnull().any().any()
    assert len(prep.dropped_cols_) == 1

def test_encode_categorical(sample_df):
    prep = DataPreprocessor(sample_df)
    prep.remove_missing(0.6)
    df2 = prep.encode_categorical()
    assert 'cat1_A' in df2.columns
    assert df2.shape[1] > sample_df.shape[1]

def test_normalize(sample_df):
    prep = DataPreprocessor(sample_df)
    prep.remove_missing(0.6)
    prep.encode_categorical()
    df3 = prep.normalize_numeric('minmax')
    num_cols = ['num1', 'num2', 'good_num']
    for col in num_cols:
        if col in df3.columns:
            assert df3[col].min() >= 0 and df3[col].max() <= 1

def test_fit_transform(sample_df):
    prep = DataPreprocessor(sample_df)
    df_clean = prep.fit_transform(threshold=0.5, method='std')
    assert prep.transformed_df_.equals(df_clean)
    assert not df_clean.isnull().any().any()

def test_invalid_method(sample_df):
    prep = DataPreprocessor(sample_df)
    with pytest.raises(ValueError):
        prep.normalize_numeric('invalid')

def test_transform(sample_df):
    prep = DataPreprocessor(sample_df)
    prep.fit_transform()
    new_df = sample_df.copy()
    df_trans = prep.transform(new_df)
    assert df_trans.shape == prep.transformed_df_.shape
    assert not df_trans.isnull().any().any().any()

def test_transform_unfitted():
    prep = DataPreprocessor(pd.DataFrame())
    with pytest.raises(ValueError):
        prep.transform(pd.DataFrame())


