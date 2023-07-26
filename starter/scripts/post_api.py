import requests
import json

# Your Heroku app's URL
url = "https://udacity-census-pred-968dfa043d2f.herokuapp.com/predict"

# The data to send
data = {
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

# Convert data to JSON format
data_json = json.dumps(data)

# Send a POST request
response = requests.post(url, data=data_json, headers={'Content-Type': 'application/json'})

# Print the HTTP status code and the response
print(f"HTTP Status Code: {response.status_code}")
print(f"Response: {response.json()}")
