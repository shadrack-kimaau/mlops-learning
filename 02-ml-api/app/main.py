from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("app/model.pkl")

app = FastAPI()

class IrisInput(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "ML API running"}

@app.post("/predict")
def predict(data: IrisInput):

    prediction = model.predict(
        np.array(data.features).reshape(1, -1)
    )

    return {
        "prediction": int(prediction[0])
    }