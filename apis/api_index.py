from fastapi import FastAPI, HTTPException
import traceback

from services.data_service import get_dataset
from services.index_service import (
    build_and_filter_index,
    clear_index_cache
)

app = FastAPI(
    title="Index Management API"
)


@app.post("/rebuild-index")
def rebuild_index():

    try:

        print("[INDEX] Loading dataset...")

        dataset = get_dataset()

        print("[INDEX] Rebuilding inverted index...")

        build_and_filter_index(
            dataset=dataset,
            max_docs=None,      
            min_df=2,
            max_df_ratio=0.9
        )

        clear_index_cache()

        print("[INDEX] Rebuild completed.")

        return {
            "status": "success",
            "message": "Index rebuilt successfully."
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    

    #uvicorn apis.api_index:app --port 8010 --reload