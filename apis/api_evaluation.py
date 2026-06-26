from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import traceback

from evaluation.evaluate_system import run_evaluation_suite

app = FastAPI(
    title="IR System Evaluation API"
)


class EvaluationResponse(BaseModel):
    status: str
    metrics: dict


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["Evaluation"]
)
def evaluate_models(
        model: str = Query(
            ...,
            description="""
            Available models:
            bm25
            tfidf
            hybrid_parallel
            hybrid_serial
            """
        )
):

    try:

        results = run_evaluation_suite(
            target_model=model
        )

        return {
            "status": "success",
            "metrics": results
        }

    except Exception as e:

        print("\n========== FULL ERROR ==========")
        traceback.print_exc()
        print("================================\n")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )