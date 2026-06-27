from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# FastAPI app
app = FastAPI(title="FastAPI + Logistic Regression Iris Predictor")

# Load scaler and logistic regression model once at startup
scaler = joblib.load("iris_scaler.joblib")
model = joblib.load("iris_model.joblib")

class_names = ["setosa", "versicolor", "virginica"]

class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs to try the predictor."}

@app.post("/predict")
def predict(req: IrisRequest):
    # Convert request to model input
    x = [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]
    x_scaled = scaler.transform(x)

    # Run logistic regression model
    probs = model.predict_proba(x_scaled)[0]
    pred_idx = int(model.predict(x_scaled)[0])

    # Return JSON response
    return {
        "predicted_class": class_names[pred_idx],
        "class_index": pred_idx,
        "probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        },
    }

