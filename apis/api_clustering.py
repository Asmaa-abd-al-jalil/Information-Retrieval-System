from fastapi import FastAPI
from pydantic import BaseModel
from services.ClusteringService import ClusteringService

app = FastAPI()
service = ClusteringService(n_clusters=3)

class ClusterRequest(BaseModel):
    documents: list[str]

@app.post("/cluster")
def cluster_docs(request: ClusterRequest):
    return service.cluster(request.documents)

#    uvicorn apis.api_clustering:app --port 8005 --reload 