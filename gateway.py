from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

REFINEMENT_URL = "http://127.0.0.1:8004/refine"
RETRIEVAL_URL = "http://127.0.0.1:8002/retrieve"
RANKING_URL = "http://127.0.0.1:8003/rank"
CLUSTERING_URL = "http://127.0.0.1:8005/cluster"
CRAWLING_URL = "http://127.0.0.1:8007/start-crawl"

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ClusteringRequest(BaseModel):
    documents: list[str]

class CrawlRequest(BaseModel):
    url: str

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

@app.post("/cluster-documents")
def cluster_gateway(request: ClusteringRequest):

    response = requests.post(CLUSTERING_URL, json={"documents": request.documents})
    
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Clustering service is unreachable")
        
    return response.json()


@app.post("/crawl")
def gateway_crawl(request: CrawlRequest):
    response = requests.post(CRAWLING_URL, json=request.dict())
    return response.json()