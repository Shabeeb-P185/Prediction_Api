from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Order Value Prediction App")

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Set up Jinja2 template directory
templates = Jinja2Templates(directory="templates")

# Home page UI
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Handle form submission
@app.post("/", response_class=HTMLResponse)
def post_form(request: Request, age: int = Form(...), gender: str = Form(...)):
    gender_val = 1 if gender.lower() == "male" else 0
    features = np.array([[age, gender_val]])
    prediction = round(model.predict(features)[0], 2)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": prediction
    })
