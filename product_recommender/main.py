from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and encoders
model = joblib.load("best_classification_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_category = joblib.load("le_category.pkl")

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, age: int = Form(...), gender: str = Form(...)):
    try:
        gender_clean = gender.strip().capitalize()
        gender_encoded = le_gender.transform([gender_clean])[0]
        features = pd.DataFrame([[age, gender_encoded]], columns=["Customer Age", "Customer Gender"])

        prediction = model.predict(features)
        category = le_category.inverse_transform(prediction)[0]
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": category
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": f"Error: {str(e)}"
        })
