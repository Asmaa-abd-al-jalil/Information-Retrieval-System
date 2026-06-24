from fastapi import FastAPI
from pydantic import BaseModel
from services.CrawlingService import CrawlingService

app = FastAPI()
crawler = CrawlingService()

class CrawlRequest(BaseModel):
    url: str

@app.post("/start-crawl")
def start_crawl(request: CrawlRequest):
    return crawler.crawl(request.url)


    # uvicorn apis.api_crawling:app --port 8007 --reload