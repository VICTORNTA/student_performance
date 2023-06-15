from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/home")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predictdata")
async def predict_datapoint(request: Request):
    form_data = await request.form()
    data = CustomData(
        gender=form_data.get('gender'),
        race_ethnicity=form_data.get('ethnicity'),
        parental_level_of_education=form_data.get('parental_level_of_education'),
        lunch=form_data.get('lunch'),
        test_preparation_course=form_data.get('test_preparation_course'),
        reading_score=float(form_data.get('writing_score')),
        writing_score=float(form_data.get('reading_score'))
    )
    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")
    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("after Prediction")
    return templates.TemplateResponse("home.html", {"request": request, "results": results[0]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")
