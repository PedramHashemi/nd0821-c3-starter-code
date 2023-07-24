"""Testing the API."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_locally_get_root():
    """Test if the status is OK."""

    response = client.get("/")
    assert response.status_code == 200

def test_prediction_less_than_50K():
    """ Test an example when income is less than 50K """

    response = client.post("/inference/", json={
        "age": 20,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
        })
    assert response.status_code == 200
    assert response.json() == {"salary": "<=50K"}

def test_prediction_more_than_50k():
    """ Test an example when income is less than 50K """

    response = client.post("/inference/", json={
            "age": 44,
            "workclass": "Private",
            "fnlgt": 167005,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 7688,
            "capital_loss": 0,
            "hours_per_week": 60,
            "native_country": "United-States"
        })
    assert response.status_code == 200
    assert response.json() == {"salary": ">50K"}
