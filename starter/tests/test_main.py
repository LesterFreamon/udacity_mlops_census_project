from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the model API"}


def test_predict_below_50k():
    data = {  # Replace with an example of data that would predict a result below 50k
        "age": 31,
        "workclass": "private",
        "education": "bachelors",
        "marital-status": "married_civ_spouse",
        "occupation": "sales",
        "relationship": "husband",
        "race": "white",
        "sex": "male",
        "native-country": "united_states",
        "fnlgt": 45781,
        "education-num": 13,
        "capital-gain": 0,
        "hours-per-week": 50,
        "capital-loss": 0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": [0]}  # Assuming your model returns [0] for incomes below 50k


def test_predict_above_50k():
    data = {  # Replace with an example of data that would predict a result above 50k
        "age": 41,
        "workclass": "private",
        "education": "masters",
        "marital-status": "married_civ_spouse",
        "occupation": "exec-managerial",
        "relationship": "husband",
        "race": "white",
        "sex": "male",
        "native-country": "united_states",
        "fnlgt": 45781,
        "education-num": 15,
        "capital-gain": 5000,
        "hours-per-week": 60,
        "capital-loss": 0
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": [1]}  # Assuming your model returns [1] for incomes above 50k
