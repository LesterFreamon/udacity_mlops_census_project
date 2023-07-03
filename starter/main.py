import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .src.config import CAT_FEATURES

app = FastAPI()


class InputData(BaseModel):
    """Use this data model to parse the input data JSON request body."""
    age: int = Field(..., example=31)
    workclass: str = Field(..., example="private", alias="work-class")
    education: str = Field(..., example="bachelors")
    marital_status: str = Field(..., example="married_civ_spouse", alias="marital-status")
    occupation: str = Field(..., example="sales")
    relationship: str = Field(..., example="husband")
    race: str = Field(..., example="white")
    sex: str = Field(..., example='male')
    native_country: str = Field(..., example="united_states", alias="native-country")
    fnlgt: int = Field(..., example=45781)
    education_num: int = Field(..., example=13, alias="education-num")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    hours_per_week: int = Field(..., example=50, alias="hours-per-week")
    capital_loss: int = Field(..., example=0, alias="capital-loss")

    class Config:
        allow_population_by_field_name = True


# Load the trained model and encoder when the application starts

try:
    with open('./model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    model = None
    encoder = None


@app.get("/")
async def root():
    return {"message": "Welcome to the model API"}


@app.post("/predict")
async def predict(data: InputData):

    if model is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model or encoder not loaded")

    # Prepare the data for model
    data_dict = data.dict(by_alias=True)

    # Transform the data_dict into a DataFrame for the encoder
    data_df = pd.DataFrame([data_dict])

    # Apply encoder transformations to the appropriate columns
    data_encoded = encoder.transform(data_df[CAT_FEATURES])

    # Concatenate the encoded columns with the non-encoded columns
    non_encoded_cols = data_df.columns.difference(CAT_FEATURES)
    data_encoded = pd.concat([data_df[non_encoded_cols], data_encoded], axis=1)

    # Model inference
    prediction = model.predict(data_encoded)

    return {"prediction": prediction.tolist()}  # jsonify the numpy array before returning
