from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the saved KMeans model and LabelEncoder
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# encoding function 
def encode_gender(gender):
    return 1 if gender.lower() == 'male' else 0  # 1: Male, 0: Female 

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def predict_cluster(request: Request, aov: float = Form(...), age: int = Form(...), gender: str = Form(...)):
    try:
        gender_encoded = encode_gender(gender)
        features = np.array([[aov, age, gender_encoded]])
        cluster = int(kmeans.predict(features)[0])
        return templates.TemplateResponse("form.html", {"request": request, "result": cluster})
    except Exception as e:
        return templates.TemplateResponse("form.html", {"request": request, "error": str(e)})
