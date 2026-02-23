import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
import re

# Ensure the root directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from search.searxng_client import SearxngClient
from llm.groq_engine import GroqEngine
from config.settings import get_settings
from ui.components.styles import apply_styles
from ui.components.results import render_source_cards, render_news_section, render_image_gallery

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EGO Sovereign AI",
    page_icon="🛡️",
    layout="wide",
)

# Apply global styles
apply_styles()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "has_searched" not in st.session_state:
        st.session_state.has_searched = False

def process_citations(text: str) -> str:
    """
    Converts [n] and [n, m] citations into circular HTML elements.
    """
    # Pattern for [1], [2, 3], etc.
    def replace_citation(match):
        nums = match.group(1).split(",")
        html = ""
        for n in nums:
            n = n.strip()
            html += f'<sup class="citation">{n}</sup>'
        return html
        
    return re.sub(r'\[(\d+(?:\s*,\s*\d+)*)\]', replace_citation, text)

initialize_session_state()

# --- HEADER ---
# Optional: Space for a custom logo or menu
st.empty()

# --- HERO SECTION ---
if not st.session_state.has_searched:
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title-primary">EGO Sovereign AI</h1>
            <h1 class="hero-title-secondary">Enterprise Sovereignty</h1>
            <p class="hero-subtitle">Secure data search with AI-powered enterprise intelligence</p>
        </div>
    """, unsafe_allow_html=True)

# --- CHAT / SEARCH AREA ---
container = st.container()

with container:
    if st.session_state.has_searched:
        # Split layout: Main Chat (3) + Right Sidebar (1)
        main_col, side_col = st.columns([3, 1], gap="large")
        
        with main_col:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if "sources" in msg:
                        render_source_cards(msg["sources"])
                    
                    if msg["role"] == "assistant":
                         st.markdown("""
                            <div class="answer-title-container">
                                <svg class="answer-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path></svg>
                                <span class="answer-label">Answer</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Process citations before rendering
                    processed_content = process_citations(msg["content"])
                    st.markdown(processed_content, unsafe_allow_html=True)
        
        with side_col:
            # Show news/images for the latest message
            if st.session_state.messages:
                last_msg = st.session_state.messages[-1]
                if "images" in last_msg:
                    render_image_gallery(last_msg["images"])
                if "news" in last_msg:
                    render_news_section(last_msg["news"])

# --- INPUT HANDLING ---
# This is tricky in Streamlit. We want it at the bottom.
# Using a fixed-height padding at the bottom for safety.
st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask anything..."):
    st.session_state.has_searched = True
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Processing the search (if we just added a user message)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    # Trigger search/LLM (Mocked or real logic here)
    # Since st.rerun happened, we need to handle the search phase.
    # To keep it reactive, we'll run it within the assistant block.
    
    with st.chat_message("assistant"):
        with st.status("🔍 Searching...", expanded=True) as status:
            search_client = SearxngClient()
            # Note: We need news and images too if possible.
            # For now, let's treat the returned results.
            sources = asyncio.run(search_client.search(user_query))
            status.update(label="Analyzing sources...", state="running")
            
            # Context for LLM
            context = "\n\n".join([f"[{i+1}] {s['title']}\n{s['content']}" for i, s in enumerate(sources)])
            
            engine = GroqEngine()
            
            status.update(label="Generating answer...", state="complete")
        
        # Display Sources first
        render_source_cards(sources)
        
        st.markdown("""
            <div class="answer-title-container">
                <svg class="answer-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path></svg>
                <span class="answer-label">Answer</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Stream response
        full_response = ""
        placeholder = st.empty()
        for chunk in engine.stream_answer(user_query, context):
            full_response += chunk
            # Process citations during streaming for real-time preview
            # Note: partial citations like [1 might flicker, so we only process complete ones
            processed_stream = process_citations(full_response)
            placeholder.markdown(processed_stream, unsafe_allow_html=True)
        
        # Update history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "images": [], # Add if handled
            "news": []    # Add if handled
        })
        st.rerun() # Final rerun to settle layout
