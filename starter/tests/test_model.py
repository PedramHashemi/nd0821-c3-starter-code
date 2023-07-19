"""Test for model functions."""

from ml.model import inference, train_model, compute_model_metrics
from ml.data import process_data
import pytest
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

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

@pytest.fixture(name='data')
def data():
    """
    Census data as fixture.
    """
    return pd.read_csv("../starter/data/census.csv")

@pytest.fixture(name='lb')
def lb():
    """
    label binerizer for the preprocessing the data.
    """
    return joblib.load("starter/ml/model/lb.pkl")

@pytest.fixture(name='encoder')
def encoder():
    """
    encoder for the preprocessing the data.
    """
    return joblib.load("starter/ml/model/encoder.pkl")

@pytest.fixture(name='sample')
def sample():
    """
    Sample data for inference.
    """
    return data.iloc[0]

def test_train_model():
    """
    Test if the model is a random forest.
    """

    model = joblib.load("starter/ml/model/model.pkl")
    assert isinstance(model, RandomForestClassifier)

def test_inference(sample):
    """Generate the inference on a sample."""
    model = joblib.load("starter/ml/model/model.pkl")
    x_sample, _, _, _ = process_data(
        sample,
        cat_features,
        label='salary',
        encoder=encoder,
        lb=lb
    )
    assert round(inference(model, x_sample)).isin([0, 1])

def test_compute_model_metrics():
    """Test the metrics."""
    model = joblib.load("starter/ml/model/model.pkl")
    x_sample, label, _, _ = process_data(
        sample,
        cat_features,
        label='salary',
        encoder=encoder,
        lb=lb
    )
    pred = round(inference(model, x_sample))
    metrics = compute_model_metrics(label, pred)
    assert len(metrics) == 3
    assert metrics[0] <= 1
    assert metrics[1] <= 1
    assert metrics[2] <= 1
