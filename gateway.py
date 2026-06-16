from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

# عناوين الخدمات
REFINEMENT_URL = "http://127.0.0.1:8004/refine"
RETRIEVAL_URL = "http://127.0.0.1:8002/retrieve"
RANKING_URL = "http://127.0.0.1:8003/rank"

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_gateway(request: SearchRequest):
    ref_resp = requests.post(REFINEMENT_URL, json={"query": request.query}).json()
    refined_query = ref_resp["refined"]

    ret_resp = requests.post(RETRIEVAL_URL, json={"query": refined_query, "top_k": request.top_k * 2}).json()
    retrieved_docs = ret_resp["results"] 
    retrieved_ids = [doc[0] for doc in retrieved_docs]

    rank_resp = requests.post(RANKING_URL, json={
        "query_tokens": refined_query.split(),
        "retrieved_doc_ids": retrieved_ids
    }).json()

    return {
        "original_query": request.query,
        "refined_query": refined_query,
        "final_results": rank_resp["ranked_results"]
    }