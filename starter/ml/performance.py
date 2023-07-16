import os
import pandas as pd
import joblib
from data import process_data
from model import inference, compute_model_metrics
import pprint
import json

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

def performance(slice_size: float=.2):
    """Get the performance on slices"""

    data = pd.read_csv("starter/data/census.csv")
    model = joblib.load("starter/model/model.pkl")
    lb = joblib.load("starter/model/lb.pkl")
    encoder = joblib.load("starter/model/encoder.pkl")

    df_sample = data.sample(frac=slice_size, replace=False)
    metrics = []

    for ft in cat_features:
        unique_vals = df_sample[ft].unique()
        for val in unique_vals:
            df_slice = df_sample[df_sample[ft] == val]
            X_slice, y_slice, encoder, lb = process_data(
                df_slice,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )
            y_pred = inference(model, X_slice)
            prcision, recall, fbeta = compute_model_metrics(y_slice, y_pred)
            slice_dict = {
                "feather": ft,
                "feature_value": val,
                "precision": prcision,
                "recall": recall,
                "fbt": fbeta,
            }
            metrics.append(slice_dict)
    return metrics

if __name__ == "__main__":
    metrics = performance(slice_size=.2)
    pprint.pprint(metrics)
    with open('starter/data/performance.json',
              'w',
              encoding='utf8') as outfile:
        outfile.write(metrics)
