import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the column names"""
    df.columns = df.columns.str.replace(' ', '')  # Delete spaces
    df.columns = df.columns.str.lower()  # Convert to lowercase
    df.columns = df.columns.str.replace('-', '_')  # Replace hyphen with underscore
    return df


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the object columns"""
    # Select columns where dtype is object
    string_columns = df.select_dtypes(include=['object']).columns

    # Apply transformations to each string column
    for col in string_columns:
        df[col] = df[col].str.replace(' ', '')  # Delete spaces
        df[col] = df[col].str.lower()  # Convert to lowercase
        df[col] = df[col].str.replace('-', '_')  # Replace hyphen with underscore

        # Convert to categorical
        df[col] = df[col].astype('category')

    return df


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans orginal file model input data."""
    clean_col_df = clean_column_names(raw_df)
    clean_df = clean_string_columns(clean_col_df)
    return clean_df
