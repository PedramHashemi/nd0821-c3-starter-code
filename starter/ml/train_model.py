"""Script to train machine learning model."""

from sklearn.model_selection import train_test_split
from model import inference, train_model, compute_model_metrics
import dvc.api
import pandas as pd
from data import process_data
import joblib


# Add code to load in the data.
# with dvc.api.open(
#     "data/census.csv"
# ) as d:
#     data = pd.read_csv(d)
data = pd.read_csv("starter/data/census.csv")

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=0)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
print(X_test)
# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, 'starter/model/model.pkl')
joblib.dump(encoder, 'starter/model/encoder.pkl')
joblib.dump(lb, 'starter/model/lb.pkl')

# Inference
y_pred = inference(model, X_test)
print(y_pred)
print(sum(y_pred))
print(test.iloc[-3])

# Performance

metrics = compute_model_metrics(y_test, y_pred)
print(metrics)
