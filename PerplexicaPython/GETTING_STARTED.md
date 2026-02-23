# Getting Started with Perplexica Python

## Installation

### 1. Create a virtual environment (recommended)

```bash
cd PerplexicaPython
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual API keys
# Required:
# - GROQ_API_KEY: Get from https://console.groq.com
# - SEARXNG_URL: Use http://localhost:8080 or a public instance

# Optional:
# - OPENWEATHER_API_KEY: For weather widget (get from https://openweathermap.org/api)
```

### 4. Initialize the database

```bash
python -c "from database.connection import init_db; init_db()"
```

### 5. Run the application

```bash
streamlit run ui/app.py
```

The app will open in your browser at `http://localhost:8501`

## Current Status

✅ **Phase 1 Complete: Foundation**
- Project structure
- Configuration system
- Type definitions
- Database models
- Session management
- Basic Streamlit UI

🚧 **In Progress: Phase 2 - LLM Integration**
- Groq provider implementation
- Model registry
- Streaming support

📋 **Upcoming:**
- Phase 3: Search layer (SearXNG)
- Phase 4: Widgets (weather, stock, calculation)
- Phase 5: Agents (classifier, researcher, writer)
- Phase 6: Full UI integration
- Phase 7: Testing & refinement

## Testing the Foundation

You can test that everything is set up correctly:

```bash
# Test imports
python -c "from config.settings import get_settings; print(get_settings())"

# Test database
python -c "from database.connection import init_db; init_db(); print('Database initialized')"

# Run the app
streamlit run ui/app.py
```

## Next Steps

1. **Get API Keys:**
   - Groq API: https://console.groq.com
   - OpenWeather (optional): https://openweathermap.org/api

2. **Set up SearXNG:**
   - Option 1: Run locally with Docker:
     ```bash
     docker run -d -p 8080:8080 searxng/searxng
     ```
   - Option 2: Use a public instance (update SEARXNG_URL in .env)

3. **Start Development:**
   - The foundation is ready!
   - Next: Implement LLM integration (Phase 2)

## Troubleshooting

### Import errors
Make sure you're in the virtual environment and have installed all dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Database errors
Initialize the database:
```bash
python -c "from database.connection import init_db; init_db()"
```

### Streamlit not found
Install streamlit:
```bash
pip install streamlit
```

## Project Structure Overview

```
PerplexicaPython/
├── config/              ✅ Settings and model configurations
├── core/                ✅ Types, utilities, session management
├── database/            ✅ SQLAlchemy models and connection
├── llm/                 🚧 LLM provider implementations (next)
├── search/              📋 SearXNG client (upcoming)
├── agents/              📋 AI agents (upcoming)
├── widgets/             📋 Specialized widgets (upcoming)
├── prompts/             📋 LLM prompts (upcoming)
├── ui/                  ✅ Streamlit app (basic skeleton)
└── tests/               📋 Test suite (upcoming)
```

Legend:
- ✅ Complete
- 🚧 In progress
- 📋 Planned
