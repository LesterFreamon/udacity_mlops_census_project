import pytest
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ..src.ml.data import process_data  # Assuming the function is in a file called your_module.py


def test_process_data_training():
    """Test process_data function in training mode."""
    X = pd.DataFrame({
        'feature1': ['a', 'b', 'a', 'b'],
        'feature2': [1, 2, 3, 4],
        'label': [1, 0, 1, 0]
    })
    X_transformed, y, encoder, lb = process_data(X, ['feature1'], 'label', training=True)
    assert X_transformed.shape == (4, 3)  # 2 unique values for feature1, and 1 continuous feature
    assert y.tolist() == [1, 0, 1, 0]  # Check if labels are same
    assert isinstance(encoder, OneHotEncoder)  # Check if encoder is instance of OneHotEncoder
    assert isinstance(lb, LabelBinarizer)  # Check if lb is instance of LabelBinarizer


def test_process_data_inference():
    """Test process_data function in inference mode."""
    X = pd.DataFrame({
        'feature1': ['a', 'b', 'a', 'b'],
        'feature2': [1, 2, 3, 4],
        'label': [1, 0, 1, 0]
    })
    X_transformed, y, encoder, lb = process_data(X, ['feature1'], 'label', training=True)
    X_new = pd.DataFrame({
        'feature1': ['a', 'b'],
        'feature2': [5, 6],
    })
    X_new_transformed, _, _, _ = process_data(X_new, ['feature1'], None, training=False, encoder=encoder, lb=lb)
    assert X_new_transformed.shape == (2, 3)  # 2 unique values for feature1, and 1 continuous feature


def test_process_data_inference_without_encoder_or_lb():
    """Test process_data function in inference mode without passing encoder or lb."""
    X_new = pd.DataFrame({
        'feature1': ['a', 'b'],
        'feature2': [5, 6],
    })
    with pytest.raises(ValueError):
        process_data(X_new, ['feature1'], None, training=False)


def test_process_data_without_categorical_features():
    """Test process_data function without categorical features."""
    X = pd.DataFrame({
        'feature1': ['a', 'b', 'a', 'b'],
        'feature2': [1, 2, 3, 4],
        'label': [1, 0, 1, 0]
    })
    X_transformed, y, encoder, lb = process_data(X, [], 'label', training=True)
    assert X_transformed.shape == (4, 2)  # Both 'feature1' and 'feature2' are treated as continuous
    assert y.shape == (4,)  # Label has 4 values
    assert encoder is None  # Check if encoder is None
    assert isinstance(lb, LabelBinarizer)  # Check if lb is instance of LabelBinarizer


def test_process_data_without_label():
    """Test process_data function without label."""
    X = pd.DataFrame({
        'feature1': ['a', 'b', 'a', 'b'],
        'feature2': [1, 2, 3, 4],
    })
    X_transformed, y, encoder, lb = process_data(X, ['feature1'], None, training=True)
    assert X_transformed.shape == (4, 3)
    assert y.size == 0  # No labels, y should be empty
    assert isinstance(encoder, OneHotEncoder)  # Check if encoder is instance of OneHotEncoder
    assert lb is None  # No LabelBinarizer needed as no labels
