"""Data processing functions."""
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X: pd.DataFrame,
    categorical_features: List[str] = [],
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, Optional[LabelBinarizer]]:
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if not training and ((encoder is None) or (lb is None)):
        raise ValueError("encoder and lb must be provided for inference")

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
        if training is True:
            lb = LabelBinarizer()
            y = lb.fit_transform(y.values).ravel()
            y = pd.Series(y, name=label)
        elif isinstance(lb, LabelBinarizer):
            try:
                y = lb.transform(y.values).ravel()
                y = pd.Series(y, name=label)
            # Catch the case where y is None because we're doing inference.
            except AttributeError:
                pass
    else:
        y = pd.Series()

    if len(categorical_features) > 0:
        X_categorical = X[categorical_features]
        if training is True:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X_categorical = encoder.fit_transform(X_categorical)
            categorical_columns = encoder.get_feature_names(categorical_features)
        elif isinstance(encoder, OneHotEncoder):
            X_categorical = encoder.transform(X_categorical)
            categorical_columns = encoder.get_feature_names(categorical_features)
        else:
            categorical_columns = []
        X_categorical = pd.DataFrame(X_categorical, columns=categorical_columns, index=X.index)
    else:
        X_categorical = pd.DataFrame()

    X_continuous = X.drop(categorical_features, axis=1)

    X = pd.concat([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
