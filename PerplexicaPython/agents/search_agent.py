import asyncio
from typing import List, Dict, Any, Optional, Generator
from search.serper_client import SerperClient
from search.jina_scraper import JinaScraper
from core.content_selection import select_relevant_content
from llm.groq_engine import GroqEngine
from core.ticker_detection import detect_company_ticker

class SearchAgent:
    """
    Orchestrates the search, scrape, and answer generation process.
    """
    
    def __init__(self):
        self.search_client = SerperClient()
        self.scraper = JinaScraper()
        self.llm = GroqEngine()
        
    async def run(self, query: str, history: List[Dict[str, str]] = []):
        """
        Runs the full pipeline and yields status and data updates.
        This is designed to be consumed by a Streamlit UI.
        """
        yield {"type": "status", "message": "Searching for sources..."}
        
        # 1. Search (Web, News, Images in parallel)
        web_task = self.search_client.search(query, "search", limit=6)
        news_task = self.search_client.search(query, "news", limit=5)
        images_task = self.search_client.search(query, "images", limit=6)
        
        web_results, news_results, image_results = await asyncio.gather(web_task, news_task, images_task)
        
        yield {
            "type": "sources", 
            "sources": web_results, 
            "news": news_results, 
            "images": image_results
        }
        
        # 2. Ticker Detection
        ticker = detect_company_ticker(query)
        if ticker:
            yield {"type": "ticker", "symbol": ticker}
            
        yield {"type": "status", "message": "Scraping and analyzing content..."}
        
        # 3. Scrape top web results
        urls_to_scrape = [res["url"] for res in web_results]
        scraped_data = await self.scraper.scrape_batch(urls_to_scrape)
        
        # 4. Process context
        context_parts = []
        for i, res in enumerate(web_results):
            # Find corresponding markdown
            md = next((s["markdown"] for s in scraped_data if s["url"] == res["url"]), "")
            content = md if md else res.get("snippet", "")
            
            relevant = select_relevant_content(content, query, 2000)
            context_parts.append(f"[{i+1}] {res['title']}\nURL: {res['url']}\n{relevant}")
            
            # Update web_results with character count info for UI
            res["content_length"] = len(content)
            
        context = "\n\n---\n\n".join(context_parts)
        
        # Re-yield sources with content length updates
        yield {
            "type": "sources", 
            "sources": web_results, 
            "news": news_results, 
            "images": image_results
        }
        
        yield {"type": "status", "message": "Generating answer..."}
        
        # 5. Stream LLM Answer
        # Note: Streaming in Python is tricky with async generators, 
        # but for Streamlit we can use sync generators or specific yielding.
        # We'll use a wrapper to make it feel smooth.
        
        def stream_wrapper():
            return self.llm.stream_answer(query, context, history)
            
        yield {"type": "answer_stream", "stream": stream_wrapper()}
        
        # 6. Follow-ups (non-streaming, at the end)
        # We need the full answer to generate follow-ups
        # The UI will handle collecting the full answer from the stream.
        # For this orchestrator, we'll return a helper to generate them later.
        yield {"type": "ready_for_followups"}
        
    def get_follow_ups(self, query: str, full_answer: str) -> List[str]:
        return self.llm.generate_follow_ups(query, full_answer)
