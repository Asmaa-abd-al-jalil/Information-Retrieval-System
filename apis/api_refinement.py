from fastapi import FastAPI, Body
from query.query_refinement_service import correct_query

app = FastAPI()

@app.post("/refine")
def refine(query: str = Body(..., embed=True)):
    corrected = correct_query(query)
    return {"original": query, "refined": corrected}

#uvicorn services.api_refinement:app --port 8004 --reload