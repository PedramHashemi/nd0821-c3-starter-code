"""
Test the API Live.
"""
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

data_rich = {
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
}

data_poor = {
    "age": 38,
    "workclass": "Private",
    "fnlgt": 215646,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Divorced",
    "occupation": "Handlers-cleaners",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

r = requests.post(
    "https://census-api-7tax.onrender.com/inference/",
    json=data_poor
)
logging.info("Testing the POST request for < 50K")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")

r = requests.post(
    "https://census-api-7tax.onrender.com/inference/",
    json=data_rich
)
logging.info("Testing the POST request for >50K")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")

r = requests.get(
    "https://census-api-7tax.onrender.com/"
)

logging.info("Testing the GET request")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")