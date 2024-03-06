import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer


df = pd.read_csv(r'data.csv')

def check_float(df, column):
    if df[column].dtype != np.float64:
        return False
    return df[column].apply(lambda x: not pd.isnull(x) and x != int(x)).any()


def columns_with_missing_values(df):
    """
    Returns a list of tuples containing the column names and their data types for columns with missing values.

    Parameters:
        df (pandas DataFrame): The DataFrame to check for missing values.

    Returns:
        list: A list of tuples containing the column names and their data types.
    """
    missing_value_columns = df.columns[df.isnull().any()].tolist()
    missing_value_column_types = df[missing_value_columns].dtypes.astype(str).to_list()
    for i, col in enumerate(missing_value_columns):
        if missing_value_column_types[i] == 'float64' and not check_float(df, col):
            missing_value_column_types[i] = 'int64'
    print('Columns with missing values:\n', json.dumps(dict(zip(missing_value_columns, missing_value_column_types)), indent=4))
    return list(zip(missing_value_columns, missing_value_column_types))


def handle_missing_data(df, missing_cols, methods=None, column=None):
    """
    Handle missing data in a pandas DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame to handle missing data.
        missing_cols (list): A list of tuples containing the column names and their data types (retreived from columns_with_missing_values function).
        methods (list): A list of methods to handle missing data. If None, the default method is 'mean'.
            default method: mean
            available methods: mean, median, most_frequent, constant, knn, iterative, missing_indicator, remove
        column (str): The name of the column to add a missing indicator. If None, no missing indicator will be added.
    """
    supported_methods = ['mean', 'median', 'most_frequent', 'constant', 'knn', 'iterative', 'missing_indicator', 'remove']

    if methods is not None and len(missing_cols) != len(methods):
        raise ValueError("The number of methods does not match the number of columns with missing values.")

    methods = ['mean' for _ in range(len(missing_cols))] if methods is None else methods

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a pandas DataFrame.")

    if df.empty:
        raise ValueError("The DataFrame is empty.")

    for i, (col, dtype) in enumerate(missing_cols):
        method = methods[i]

        if col not in df.columns:
            raise ValueError(f"The column '{col}' does not exist in the DataFrame.")

        if method == 'remove':
            df = df.dropna(subset=[col])
            continue

        if dtype in ['int64', 'float64']:
            if method in ['mean', 'median', 'most_frequent', 'constant']:
                imputer = SimpleImputer(strategy=method)
                df[col] = imputer.fit_transform(df[[col]])
            elif method == 'knn':
                imputer = KNNImputer(n_neighbors=2)
                df[col] = imputer.fit_transform(df[[col]])
            elif method == 'iterative':
                imputer = IterativeImputer(max_iter=10, random_state=0)
                df[col] = imputer.fit_transform(df[[col]])
            elif method == 'missing_indicator':
                if column:
                    df[column+'_missing'] = df[column].isnull()
                else:
                    raise ValueError("Please provide a column name.")
            else:
                raise ValueError(f'Invalid method: {method}\nSupported methods: {supported_methods}')
            if dtype == 'int64':
                df[col] = df[col].astype(int)
            elif dtype == 'float64':
                df[col] = df[col].astype(float)
        elif dtype == 'object':
            if method == 'categorical':
                df[col] = df[col].fillna(df[col].value_counts().index[0])
            elif method == 'missing_indicator':
                if column:
                    df[column+'_missing'] = df[column].isnull()
                else:
                    raise ValueError("Please provide a column name.")
    return df


print('DATAFRAME BEFORE\n', df, '\n\n')

df = handle_missing_data(df, columns_with_missing_values(df))

print('\n\nDATAFRAME AFTER\n', df)
