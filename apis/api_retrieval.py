from fastapi import FastAPI
from pydantic import BaseModel
import requests 
from services.retrieval_service import retrieve

app = FastAPI()

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/retrieve")
def get_results(request: RetrievalRequest):
    response = requests.post(
        "http://127.0.0.1:8001/preprocess", 
        json={"text": request.query}
    )
    preprocessed = response.json()
    
    clean_query = preprocessed.get('final_text', request.query)
    
    results = retrieve(clean_query, top_k=request.top_k)
    return {"query": request.query, "results": results}

#  uvicorn apis.api_retrieval:app --port 8002 --reload