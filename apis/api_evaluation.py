from fastapi import FastAPI, Query
from pydantic import BaseModel
from evaluation.evaluate_system import run_evaluation_suite

app = FastAPI(title="IR System Evaluation API")

class EvaluationResponse(BaseModel):
    status: str
    metrics: dict

@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
def evaluate_models(
    model: str = Query(..., description="اختر النموذج للتقييم: tfidf, bm25, or hybrid")
):
    results = run_evaluation_suite(target_model=model) 
    return {"status": "success", "metrics": results}