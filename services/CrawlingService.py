import requests
from bs4 import BeautifulSoup

class CrawlingService:
    def crawl(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # استخراج النصوص من جميع فقرات <p>
                text = " ".join([p.get_text() for p in soup.find_all('p')])
                return {"url": url, "content": text, "status": "success"}
            
            return {"url": url, "status": "failed", "error": f"Status code {response.status_code}"}
        
        except Exception as e:
            return {"url": url, "status": "failed", "error": str(e)}