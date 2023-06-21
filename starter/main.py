# Put the code for your API here.
from typing import List

from fastapi import FastAPI
# pylint: disable=E0611
from pydantic import BaseModel, Field

app = FastAPI()


class InputData(BaseModel):
    age: int = Field(..., example=31)

@app.get("/")
async def root():
    return {"message": "Welcome to the model API"}

@app.post("/predict")
async def predict(data: InputData):
    # Replace with your model inference code
    prediction = infer_from_model(data.field_with_hyphen)

    return {"prediction": prediction}


def infer_from_model(data):
    # This is a placeholder function. Replace with your actual model inference code.
    return sum(data) / len(data)
