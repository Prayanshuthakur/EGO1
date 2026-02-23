import streamlit as st

def get_custom_css():
    """ Returns the exact CSS to match Fireplexity v2 aesthetics. """
    return """
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600&display=swap');

    :root {
        --primary: #ff4d00;
        --secondary: #6366f1;
        --bg-main: #ffffff;
        --bg-glass: rgba(255, 255, 255, 0.8);
        --border-glass: rgba(0, 0, 0, 0.08);
        --text-main: #18181b; /* Zinc 900 */
        --text-muted: #71717a; /* Zinc 500/400 */
    }

    [data-theme="dark"], .dark-theme {
        --bg-main: #09090b; /* Zinc 950/Black */
        --bg-glass: rgba(9, 9, 11, 0.8);
        --border-glass: rgba(255, 255, 255, 0.1);
        --text-main: #f4f4f5; /* Zinc 100 */
        --text-muted: #a1a1aa; /* Zinc 400 */
    }

    /* Global Styles */
    .stApp {
        background-color: transparent;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: var(--text-main);
    }

    /* Hero Typography */
    .hero-container {
        text-align: center;
        padding: 4rem 1rem 2rem 1rem;
    }
    
    .hero-title-primary {
        font-size: 3rem;
        line-height: 1.1;
        font-weight: 500;
        color: var(--primary);
        font-family: 'Outfit', sans-serif;
    }
    
    .hero-title-secondary {
        font-size: 3rem;
        line-height: 1.1;
        font-weight: 500;
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
        margin-top: -0.5rem;
    }

    .hero-subtitle {
        margin-top: 1rem;
        font-size: 1.125rem;
        color: var(--text-muted);
    }

    /* Source Cards Grid (5 columns implementation) */
    .source-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin-bottom: 2rem;
    }

    .source-card {
        background: var(--bg-glass);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border-glass);
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none !important;
        color: inherit !important;
        height: 112px; /* h-28 equivalent */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .source-card:hover {
        transform: translateY(-2px);
        border-color: #fdba74; /* orange-300 */
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    .source-header {
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }

    .source-favicon {
        width: 14px;
        height: 14px;
        border-radius: 2px;
        object-fit: contain;
    }

    .source-site-name {
        font-size: 10px;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: lowercase;
    }

    .source-title {
        font-size: 0.75rem;
        font-weight: 500;
        line-height: 1.25;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin: 0.25rem 0;
    }

    .source-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .source-char-counter {
        font-size: 9px;
        color: #9ca3af;
    }

    /* Answer Styling */
    .answer-title-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }
    
    .answer-icon {
        color: var(--text-main);
    }

    .answer-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-muted);
    }

    /* Chat Messages */
    .stChatMessage {
        background: transparent !important;
        border-bottom: 1px solid var(--border-glass) !important;
        padding: 1.5rem 0 !important;
    }

    /* Animations */
    @keyframes fade-up {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-up {
        animation: fade-up 0.5s ease-out forwards;
    }

    /* Sidebars */
    .sidebar-section-title {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-muted);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Citations Styling */
    .citation {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #f4f4f5;
        color: #71717a;
        font-size: 0.65rem;
        font-weight: 600;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin: 0 2px;
        vertical-align: super;
        text-decoration: none;
    }
    
    [data-theme="dark"] .citation, .dark-theme .citation {
        background: #27272a;
        color: #a1a1aa;
    }

    .citation:hover {
        background: var(--primary);
        color: white;
    }

    /* Hide Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

def apply_styles():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
