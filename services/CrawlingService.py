import requests
import time
from bs4 import BeautifulSoup

class CrawlingService:
    def __init__(self, delay=2):
        self.delay = delay 

    def crawl(self, url):
        try:
            headers = {
                'User-Agent': 'MyBot/1.0 (Contact: asmaamh@gmail.com)'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            time.sleep(self.delay)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = " ".join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3'])])
                return {"url": url, "content": text, "status": "success"}
            
            return {"url": url, "status": "failed", "error": f"HTTP {response.status_code}"}
        
        except Exception as e:
            return {"url": url, "status": "failed", "error": str(e)}

crawler = CrawlingService(delay=3) 
# crawler.crawl("https://example.com")