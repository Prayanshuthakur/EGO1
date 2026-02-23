import httpx
import os
import asyncio
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class JinaScraper:
    """
    Scraper using the Jina Reader API (r.jina.ai) to extract markdown from URLs.
    """
    
    def __init__(self, base_url: str = "https://r.jina.ai"):
        self.base_url = base_url.rstrip("/")
        
    async def scrape(self, url: str, timeout: int = 10) -> str:
        """
        Scrape a single URL to markdown.
        """
        scrape_url = f"{self.base_url}/{url}"
        headers = {
            "Accept": "text/plain",
            "X-Return-Format": "markdown"
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(scrape_url, headers=headers)
                if response.status_code == 200:
                    return response.text
                return ""
        except Exception as e:
            # Log error if needed: print(f"Error scraping {url}: {e}")
            return ""

    async def scrape_batch(self, urls: List[str], max_concurrent: int = 5) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs in parallel.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def sem_scrape(url):
            async with semaphore:
                markdown = await self.scrape(url)
                return {"url": url, "markdown": markdown}
                
        tasks = [sem_scrape(url) for url in urls]
        return await asyncio.gather(*tasks)
