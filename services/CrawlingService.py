import requests
import time
from bs4 import BeautifulSoup

from services.database_service import insert_documents


class CrawlingService:

    def __init__(self, delay=2):
        self.delay = delay
        self._cache = {}

    def crawl(self, url):

        if url in self._cache:
            return self._cache[url]

        try:
            headers = {
                'User-Agent': 'MyBot/1.0 (Contact: asmaamh@gmail.com)'
            }

            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            time.sleep(self.delay)

            if response.status_code == 200:

                soup = BeautifulSoup(
                    response.text,
                    'html.parser'
                )

                text = " ".join([
                    tag.get_text(strip=True)
                    for tag in soup.find_all(
                        ['p', 'h1', 'h2', 'h3']
                    )
                ])

                insert_documents({
                    url: text
                })

                result = {
                    "url": url,
                    "content": text,
                    "status": "success"
                }

                self._cache[url] = result

                return result

            result = {
                "url": url,
                "status": "failed",
                "error": f"HTTP {response.status_code}"
            }

            self._cache[url] = result

            return result

        except Exception as e:

            result = {
                "url": url,
                "status": "failed",
                "error": str(e)
            }

            self._cache[url] = result

            return result


crawler = CrawlingService(delay=3)

# crawler.crawl("https://example.com")