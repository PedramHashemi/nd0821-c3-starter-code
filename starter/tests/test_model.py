"""Test for model functions."""

from ml.model import inference, compute_model_metrics
import ml.data as proc
import pytest
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy.random import seed, rand, randint

seed(1)

@pytest.fixture(name='data')
def data():
    """
    Census data as fixture.
    """
    return pd.read_csv("./starter/data/census.csv")

@pytest.fixture(name='lb')
def lb():
    """
    label binerizer for the preprocessing the data.
    """
    return joblib.load("./starter/model/lb.pkl")

@pytest.fixture(name='encoder')
def encoder():
    """
    encoder for the preprocessing the data.
    """
    return joblib.load("./starter/model/encoder.pkl")

@pytest.fixture(name='sample')
def sample(encoder, lb):
    """
    Sample data for inference.
    """

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
    census = pd.read_csv("./starter/data/census.csv")
    samp = census.iloc[0:2]
    x_sample, y_sample, _, _ = proc.process_data(
        samp,
        categorical_features=cat_features,
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )
    return x_sample, y_sample

def test_train_model():
    """
    Test if the model is a random forest.
    """

    model = joblib.load("starter/model/model.pkl")
    assert isinstance(model, RandomForestClassifier)

def test_inference(sample):
    """Generate the inference on a sample."""

    x_sample, y_sample = sample
    model = joblib.load("starter/model/model.pkl")
    preds = inference(model, x_sample)
    # assert preds in [0, 1]

def test_compute_model_metrics():
    """Test the metrics."""

    length = randint(1)
    pred = np.round(rand(length))
    label = np.round(rand(length))
    fbeta, precision, recall = compute_model_metrics(label, pred)
    assert fbeta <= 1
    assert precision <= 1
    assert recall <= 1
