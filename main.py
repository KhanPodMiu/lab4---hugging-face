from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

clf = pipeline("sentiment-analysis", model=MODEL_ID)

app = FastAPI()


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/")
def root():
    return {
        "name": "Vo Le Nam Khanh Ann - 25127274",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    result = clf(req.text, truncation=True)[0]
    return {
        "text": req.text,
        "label": result["label"],
        "score": round(result["score"], 4),
    }