"""Machine learning model training and evaluation."""
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression(max_iter=1000)

    param_grid = {'C': [0.001, 0.01, 0.1, 1],
                  'penalty': ['l1'],
                  'solver': ['liblinear', 'saga']}

    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=12)

    grid.fit(X_train, y_train)

    # return the model with the best hyperparameters
    best_model = grid.best_estimator_
    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_metrics_per_category(
    model: LogisticRegression,
    test: pd.DataFrame,
    target: str,
    feature: str,
    cat_features: List[str],
    encoder: OneHotEncoder,
    lb: LabelBinarizer
) -> pd.DataFrame:
    """Compute and print precision, recall, and fbeta score for each unique value of a feature in the test data."""

    # Get unique values of the feature
    unique_values = test[feature].unique()

    # Initialize a DataFrame to store the results
    results = []

    for value in unique_values:
        slice_indices = test[feature] == value
        test_slice = test[slice_indices]

        X_test, y_true, _, _ = process_data(
            test_slice, categorical_features=cat_features, label=target, training=False, encoder=encoder, lb=lb
            )

        y_pred = model.predict(X_test)

        # Compute the metrics
        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

        # Store the results
        results.append(
            {feature: value, 'Precision': precision, 'Recall': recall, 'Fbeta': fbeta}
        )

    results_df = pd.DataFrame(results)

    return results_df
