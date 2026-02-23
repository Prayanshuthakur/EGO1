import re
from typing import List

def select_relevant_content(content: str, query: str, max_length: int = 2000) -> str:
    """
    Extracts the most relevant parts of the content based on the query.
    Ported from TypeScript version in lib/content-selection.ts.
    """
    if not content:
        return ""
        
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    if not paragraphs:
        return ""
        
    # Always include the first paragraph (introduction)
    intro = "\n\n".join(paragraphs[:2])
    
    # Extract keywords from the query
    stop_words = {'what', 'when', 'where', 'which', 'how', 'why', 'does', 'with', 'from', 'about'}
    keywords = [
        word.lower() for word in re.split(r'\s+', query)
        if len(word) > 3 and word.lower() not in stop_words
    ]
    
    # Find paragraphs that contain keywords
    # Skip intro (first 2) and potential conclusion (last 2) for scoring
    mid_paragraphs = paragraphs[2:-1] if len(paragraphs) > 3 else []
    
    scored_paragraphs = []
    for i, para in enumerate(mid_paragraphs):
        score = sum(1 for kw in keywords if kw in para.lower())
        if score > 0:
            scored_paragraphs.append({
                "text": para,
                "score": score,
                "index": i
            })
            
    # Take top 3 most relevant paragraphs
    scored_paragraphs.sort(key=lambda x: x["score"], reverse=True)
    top_paragraphs = scored_paragraphs[:3]
    
    # Restore original order
    top_paragraphs.sort(key=lambda x: x["index"])
    relevant_text = "\n\n".join([p["text"] for p in top_paragraphs])
    
    # Always include the last paragraph if it exists (conclusion)
    conclusion = paragraphs[-1] if len(paragraphs) > 2 else ""
    
    # Combine all parts
    result_parts = [intro]
    if relevant_text:
        result_parts.append(relevant_text)
    if conclusion and conclusion not in intro:
        result_parts.append(conclusion)
        
    result = "\n\n".join(result_parts)
    
    # Ensure we don't exceed max length
    if len(result) > max_length:
        result = result[:max_length - 3] + "..."
        
    return result
