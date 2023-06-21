import pandas as pd
import numpy as np
from starter.clean_data import clean_column_names, clean_string_columns, clean_data


def test_clean_column_names():
    raw_df = pd.DataFrame({
        ' Column Name ': np.arange(10),
        'Column-Name': np.arange(10, 20)
    })
    cleaned_df = clean_column_names(raw_df)
    assert 'columnname' in cleaned_df.columns
    assert 'column_name' in cleaned_df.columns


def test_clean_string_columns():
    raw_df = pd.DataFrame({
        'column1': [' Value ', 'Value-', 'VALUE ', 'vAlUe-'] * 5,
        'column2': np.arange(20)
    })
    cleaned_df = clean_string_columns(raw_df)
    assert cleaned_df['column1'].dtype.name == 'category'
    assert cleaned_df['column1'].str.contains(' ').any() is False
    assert cleaned_df['column1'].str.contains('-').any() is False
    assert cleaned_df['column1'].str.islower().all() is True


def test_clean_data():
    raw_df = pd.DataFrame({
        ' Column Name ': [' Value ', 'Value-', 'VALUE ', 'vAlUe-'] * 5,
        'Column-Name': np.arange(20)
    })
    cleaned_df = clean_data(raw_df)
    assert 'columnname' in cleaned_df.columns
    assert cleaned_df['column_name'].dtype.name == 'category'
    assert cleaned_df['column_name'].str.contains(' ').any() is False
    assert cleaned_df['column_name'].str.contains('-').any() is False
    assert cleaned_df['column_name'].str.islower().all() is True
