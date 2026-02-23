import streamlit as st

def render_source_cards(sources: list):
    """ Renders the exact 5-column grid of source cards. """
    if not sources:
        return
        
    st.markdown("""
        <div class="answer-title-container">
            <svg class="answer-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
            <span class="answer-label">Sources</span>
        </div>
    """, unsafe_allow_html=True)
    
    # We use raw HTML for the 5-column grid to ensure exact replication
    html = '<div class="source-grid">'
    for idx, source in enumerate(sources[:10]): # Limit to 10 for grid
        title = source.get("title", "Untitled")
        url = source.get("url", "#")
        site_name = source.get("siteName") or (source.get("url", "").split("//")[-1].split("/")[0].replace("www.", ""))
        favicon = source.get("favicon", "")
        content_len = len(source.get("content", ""))
        
        # Build the HTML carefully without extra indentation
        html += f'<a href="{url}" target="_blank" class="source-card animate-fade-up" style="animation-delay: {idx * 50}ms">'
        html += '<div class="source-header">'
        if favicon:
            html += f'<img src="{favicon}" class="source-favicon">'
        html += f'<span class="source-site-name">{site_name}</span></div>'
        html += f'<div class="source-title">{title}</div>'
        html += '<div class="source-footer">'
        html += f'<span class="source-char-counter">{content_len} chars</span>'
        html += f'<span class="source-char-counter">[{idx+1}]</span>'
        html += '</div></a>'
        
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

def render_news_section(news: list):
    """ Renders a news sidebar section. """
    if not news:
        return
        
    st.markdown("""
        <div class="sidebar-section-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/></svg>
            <span>News</span>
        </div>
    """, unsafe_allow_html=True)
    
    for item in news[:5]:
        title = item.get("title", "")
        url = item.get("url", "#")
        source = item.get("source", "")
        
        st.markdown(f"""
            <div style="margin-bottom: 1rem; opacity: 0; animation: fade-up 0.5s ease-out forwards;">
                <a href="{url}" target="_blank" style="text-decoration:none; color:inherit;">
                    <div style="font-size: 0.875rem; font-weight: 500; line-height: 1.25; margin-bottom: 0.25rem;">{title}</div>
                    <div style="font-size: 0.75rem; color: #71717a;">{source}</div>
                </a>
            </div>
        """, unsafe_allow_html=True)

def render_image_gallery(images: list):
    """ Renders an image sidebar section. """
    if not images:
        return
        
    st.markdown("""
        <div class="sidebar-section-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
            <span>Images</span>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    for idx, img in enumerate(images[:4]):
        with cols[idx % 2]:
            st.image(img.get("imageUrl"), use_container_width=True)
