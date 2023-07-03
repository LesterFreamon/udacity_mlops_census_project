import json
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CAT_FEATURES, LABEL
from .ml.data import process_data
from .ml.model import train_model, inference, compute_model_metrics

# Script to train machine learning model.

# Add the necessary imports for the src code.

# Add code to load in the data.
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'clean_census.csv')
print(data_path)
data = pd.read_csv(data_path)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label=LABEL, training=True
)

model_data_dir = os.path.join(current_dir, '..', 'model')
# Save the encoder and label binarizer
with open(os.path.join(model_data_dir, 'encoder.pkl'), 'wb') as f:
    pickle.dump(encoder, f)
with open(os.path.join(model_data_dir, 'lb.pkl'), 'wb') as f:
    pickle.dump(lb, f)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save the model
with open(os.path.join(model_data_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Perform inference on the test data
preds_test = inference(model, X_test)

# Compute the model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds_test)
# Save the metrics in a json file
metrics_dir = os.path.join(current_dir, '..', 'reports')
with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as file:
    json.dump({'precision': precision, 'recall': recall, 'fbeta': fbeta}, file)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F-beta score: {fbeta}')
