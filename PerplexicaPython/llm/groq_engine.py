from groq import Groq
import os
from typing import List, Dict, Any, Generator, Optional
from dotenv import load_dotenv

load_dotenv()

class GroqEngine:
    """
    Handles LLM interactions using Groq.
    Supports streaming and non-streaming responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from config.settings import get_settings
        settings = get_settings()
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in settings or environment.")
            
        self.client = Groq(api_key=self.api_key)
        
    def stream_answer(self, query: str, context: str, history: List[Dict[str, str]] = []) -> Generator[str, None, None]:
        """
        Streams an answer from Groq based on the provided context.
        """
        system_prompt = """You are a friendly assistant that helps users find information.

CRITICAL FORMATTING RULE:
- NEVER use LaTeX/math syntax ($...$) for regular numbers in your response
- Write ALL numbers as plain text: "1 million" NOT "$1$ million", "50%" NOT "$50\\%$"
- Only use math syntax for actual mathematical equations if absolutely necessary

RESPONSE STYLE:
- For greetings, respond warmly and ask how you can help
- For simple questions, give direct, concise answers
- MATCH the user's energy level
- Include citations as [1], [2], etc. when referencing sources.
- Citations correspond to the source order provided.
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history
        for msg in history:
            messages.append(msg)
            
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"Answer this query: \"{query}\"\n\nBased on these sources:\n{context}"
        })
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_follow_ups(self, query: str, answer: str) -> List[str]:
        """
        Generates follow-up questions.
        """
        prompt = f"""Generate 5 natural follow-up questions based on the query and answer.
        
Query: {query}
Answer: {answer}

Return only the questions, one per line, no numbering."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        questions = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
        return questions[:5]


