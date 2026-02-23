import httpx
from typing import List, Dict, Any, Optional
from config.settings import get_settings

class SearxngClient:
    """
    Client for SearXNG Search API.
    """
    
    def __init__(self, url: Optional[str] = None):
        settings = get_settings()
        self.url = url or settings.searxng_url
        
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a search using SearXNG.
        
        Args:
            query: The search query.
            limit: Number of results to return.
            
        Returns:
            List of search results in a normalized format.
        """
        params = {
            'q': query,
            'format': 'json',
            'engines': 'google,bing,duckduckgo',
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])
                
                return self._normalize_results(results[:limit])
        except Exception as e:
            print(f"SearXNG search error: {e}")
            return []
            
    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize SearXNG results to match our internal schema."""
        normalized = []
        for r in results:
            normalized.append({
                "url": r.get("url"),
                "title": r.get("title"),
                "content": r.get("content") or r.get("snippet", ""),
                "favicon": f"https://www.google.com/s2/favicons?domain={r.get('url')}&sz=32"
            })
        return normalized
