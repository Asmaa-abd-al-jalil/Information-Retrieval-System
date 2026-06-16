from fastapi import FastAPI
from pydantic import BaseModel
from services.preprocessing_service import preprocess_text

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/preprocess")
def preprocess(request: TextRequest):
    result = preprocess_text(request.text)
    if 'final_text' not in result:
        result['final_text'] = result.get('normalized', request.text)
    return result

# لتشغيل السيرفر: uvicorn api_preprocessing:app --reload
