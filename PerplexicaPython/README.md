# Perplexica Python

A Python implementation of Perplexica - an AI-powered search engine with Streamlit frontend.

## Features

- 🔍 **Intelligent Search**: Multi-source web search via SearXNG
- 🤖 **AI-Powered Answers**: Groq LLM integration for comprehensive responses
- 🧠 **Iterative Research**: Agent-based research with tool calling
- 🌤️ **Smart Widgets**: Weather, stock market, and calculation widgets
- 💬 **Chat Interface**: Streamlit-based conversational UI
- 📊 **Research Visualization**: See the AI's research process in real-time
- 📚 **Source Citations**: All answers include verifiable sources

## Quick Start

### Prerequisites

- Python 3.10+
- SearXNG instance (local or public)
- Groq API key

### Installation

```bash
# Clone the repository
cd PerplexicaPython

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key_here
SEARXNG_URL=http://localhost:8080
OPENWEATHER_API_KEY=your_weather_api_key  # Optional
DATABASE_URL=sqlite:///./perplexica.db
LOG_LEVEL=INFO
```

### Run

```bash
streamlit run ui/app.py
```

Open your browser to `http://localhost:8501`

## Project Structure

```
PerplexicaPython/
├── config/          # Configuration and settings
├── database/        # Database models and migrations
├── core/            # Core utilities and types
├── llm/             # LLM provider implementations
├── search/          # Search engine integration
├── agents/          # AI agents (classifier, researcher, writer)
├── widgets/         # Specialized widgets (weather, stocks, etc.)
├── prompts/         # LLM prompts
├── ui/              # Streamlit UI components
└── tests/           # Test suite
```

## Architecture

The system consists of 8 layers:

1. **Foundation**: Configuration, types, utilities
2. **Database**: SQLAlchemy models for persistence
3. **LLM Integration**: Groq provider with streaming support
4. **Search**: SearXNG client for web search
5. **Widgets**: Weather, stock, calculation widgets
6. **Agents**: Classifier, Researcher, Writer agents
7. **Session Management**: Real-time state and streaming
8. **UI**: Streamlit components

## Usage Examples

### Simple Query
```
User: What is Python?
AI: [Provides answer without search if knowledge is sufficient]
```

### Search Query
```
User: Latest developments in AI in 2026
AI: [Performs web search, shows sources, generates comprehensive answer]
```

### Widget Queries
```
User: What's the weather in Tokyo?
AI: [Shows weather widget + answer]

User: Current price of TSLA stock?
AI: [Shows stock widget + answer]

User: What is 15% of 250?
AI: [Shows calculation widget + answer]
```

### Multi-turn Conversation
```
User: Who founded OpenAI?
AI: [Answer with sources]

User: When was it founded?
AI: [Understands context, reformulates query, provides answer]
```

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_classifier.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Style

```bash
# Format code
black .

# Lint
ruff check .
```

## Roadmap

- [x] Phase 1: Foundation setup
- [ ] Phase 2: LLM integration
- [ ] Phase 3: Search layer
- [ ] Phase 4: Widgets
- [ ] Phase 5: Agents
- [ ] Phase 6: UI integration
- [ ] Phase 7: Testing & refinement

### Future Features

- Document upload and search
- Image and video search
- Multiple LLM provider support (Ollama, OpenAI, etc.)
- Advanced embeddings for semantic search
- Multi-user support with PostgreSQL
- Docker deployment

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the original [Perplexica](https://github.com/ItzCrazyKns/Perplexica) by ItzCrazyKns
