from fastapi import FastAPI, Body
from rank.rank_service import rank_retrieved_documents
from services.index_service import load_index

app = FastAPI()

index, doc_lengths, doc_count = load_index()

@app.post("/rank")
def rank(
    query_tokens: list = Body(...), 
    retrieved_doc_ids: list = Body(...)
):
    results = rank_retrieved_documents(
        query_tokens, 
        {doc_id: {} for doc_id in retrieved_doc_ids}, 
        index, 
        doc_lengths, 
        doc_count
    )
    
    return {"ranked_results": results}
# uvicorn apis.api_ranking:app --reload --port=8003