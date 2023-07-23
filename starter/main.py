"""
The module containing the API.
"""

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import os
import pickle
from ml.data import process_data
from ml.model import inference
import joblib
import uvicorn


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


class Profile(BaseModel):
    """Profile for the persona."""
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


    class Config:
        """config class."""
        schema_extra = {
            "example": {
                "age": 38,
                "workclass": 'Private',
                "fnlgt": 215646,
                "education": 'HS-grad',
                "education_num": 9,
                "marital_status": "Divorced",
                "occupation": "Handlers-cleaners",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": 'United-States',
            }
        }


app = FastAPI()

@app.get("/")
def welcome():
    """Welcome url."""
    return "Hello and Welcome to this project with FastAPI and Render."

@app.post("/inference/")
def infer(profile: Profile):
    """Infer with the given data.

    Args:
        profile (Profile): A profile of a person.
    """
    data = {
        "age": profile.age,
        "workclass": profile.workclass,
        "fnlgt": profile.fnlgt,
        "education": profile.education,
        "education_num": profile.education_num,
        "marital-status": profile.marital_status,
        "occupation": profile.occupation,
        "relationship": profile.relationship,
        "race": profile.race,
        "sex": profile.sex,
        "capital_gain": profile.capital_gain,
        "capital_loss": profile.capital_loss,
        "hours_per_week": profile.hours_per_week,
        "native-country": profile.native_country,
    }
    data_df = pd.DataFrame(data, index=[0])

    model = joblib.load("starter/model/model.pkl")
    encoder = joblib.load("starter/model/encoder.pkl")
    lb = joblib.load("starter/model/lb.pkl")

    processed_data, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    label = inference(model, processed_data)

    return {"salary": lb.inverse_transform(label)[0]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
