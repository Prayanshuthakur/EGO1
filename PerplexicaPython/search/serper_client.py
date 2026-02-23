import httpx
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class SerperClient:
    """
    Client for Serper.dev Search API.
    Provides web, news, and image search results.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev"
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment or provided.")
            
    async def search(self, query: str, search_type: str = "search", limit: int = 6) -> List[Dict[str, Any]]:
        """
        Perform a search using Serper.dev.
        
        Args:
            query: The search query.
            search_type: "search" (web), "news", or "images".
            limit: Number of results to return.
            
        Returns:
            List of search results in a normalized format.
        """
        url = f"{self.base_url}/{search_type}"
        payload = {
            "q": query,
            "num": limit
        }
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            return self._normalize_results(data, search_type)
            
    def _normalize_results(self, data: Dict[str, Any], search_type: str) -> List[Dict[str, Any]]:
        """Normalize Serper.dev results to match our internal schema."""
        results = []
        
        if search_type == "search":
            # Web results
            for item in data.get("organic", []):
                results.append({
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "siteName": item.get("source"),
                    "favicon": f"https://www.google.com/s2/favicons?domain={item.get('link')}&sz=32"
                })
        elif search_type == "news":
            # News results
            for item in data.get("news", []):
                results.append({
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "source": item.get("source"),
                    "date": item.get("date"),
                    "imageUrl": item.get("imageUrl")
                })
        elif search_type == "images":
            # Image results
            for item in data.get("images", []):
                results.append({
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "imageUrl": item.get("imageUrl"),
                    "source": item.get("source"),
                    "width": item.get("imageWidth"),
                    "height": item.get("imageHeight")
                })
                
        return results
