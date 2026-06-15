from fastapi import FastAPI
from services.index_service import build_inverted_index, save_index
from services.data_service import get_dataset

app = FastAPI()

@app.post("/build_index")
def build():
    dataset = get_dataset()
    
    inverted_index, doc_lengths, doc_count = build_inverted_index(dataset) 
    
    save_index(inverted_index, doc_lengths, doc_count)    
    
    return {"status": "success", "message": f"Index built for {doc_count} docs and saved successfully"}

#  uvicorn services.api_indexing:app --port 8002 --reload