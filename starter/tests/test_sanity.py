from fastapi.testclient import TestClient

from main import app

import json
import requests

# replace the URL with your application URL
URL = "http://localhost:5000/api"


def test_predict_endpoint():
    client = TestClient(app)
    response = client.post("/predict", json={
        "age": 31,
        "workclass": "private",
        "education": "bachelors",
        "marital_status": "married_civ_spouse",
        "occupation": "sales",
        "relationship": "husband",
        "race": "white",
        "sex": 'male',
        "native-country": "united_states",
        "fnlgt": 45781,
        "education-num": 13,
        "capital-gain": 0,
        "hours-per-week": 50,
        "capital-loss": 0
    })

    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert "prediction" in response.json(), "No prediction in the response"


def test_get_status_code():
    response = requests.get(URL)
    assert response.status_code == 200


def test_get_response_body():
    response = requests.get(URL)
    response_body = response.json()
    # replace with appropriate checks based on your expected response
    assert 'key' in response_body


def test_post_status_code():
    # replace the payload with your actual data
    payload = {"data": "data_value"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, headers=headers, data=json.dumps(payload))
    assert response.status_code == 200


def test_post_response_body_inference1():
    # replace the payload with your actual data
    payload = {"data": "data_value1"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, headers=headers, data=json.dumps(payload))
    response_body = response.json()
    # replace with appropriate checks based on your expected response
    assert response_body['result'] == 'inference1'


def test_post_response_body_inference2():
    # replace the payload with your actual data
    payload = {"data": "data_value2"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, headers=headers, data=json.dumps(payload))
    response_body = response.json()
    # replace with appropriate checks based on your expected response
    assert response_body['result'] == 'inference2'
