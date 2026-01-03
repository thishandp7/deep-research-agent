# Deep Research Assistant

A LangGraph-based research assistant that autonomously searches the web, evaluates source trustworthiness, and generates comprehensive research reports.

## Overview

This project demonstrates LangGraph fundamentals through advanced patterns by building a real-world research assistant that:

1. **Searches the web** - Generates queries and searches using DuckDuckGo
2. **Scrapes content** - Extracts main content from discovered sources
3. **Analyzes trustworthiness** - Scores sources on credibility (0-100)
4. **Stores vetted content** - Saves high-quality sources (>85%) in ChromaDB
5. **Generates reports** - Creates comprehensive HTML research reports

## Features

- **LangGraph Orchestration**: StateGraph with conditional routing
- **Open Source LLMs**: Ollama integration (llama3.2, llama3.1)
- **Vector Database**: ChromaDB for semantic search
- **Web Search**: DuckDuckGo (no API key required)
- **HTML Reports**: Professional research reports with source citations
- **Trustworthiness Scoring**: Hybrid LLM + heuristic analysis

## Project Structure

```
ResearchAssistant/
├── examples/              # LangGraph learning examples
│   ├── 01_simple_chain.py
│   ├── 02_conditional_routing.py
│   └── 03_state_management.py
├── research_assistant/    # Main package
│   ├── graph/            # LangGraph orchestration
│   ├── agents/           # Specialized agents
│   ├── tools/            # Search, scraping, vector DB
│   ├── models/           # Data models
│   └── utils/            # LLM, prompts, HTML generation
├── data/
│   ├── vector_db/        # ChromaDB storage
│   └── reports/          # Generated reports
└── tests/                # Test suite
```

## Quick Start

### 1. Setup

```bash
# Clone and navigate to project
cd ResearchAssistant

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Setup Ollama (if not already installed)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull LLM models
ollama pull llama3.2:3b    # Fast for development
ollama pull llama3.1:8b    # Better quality

# Create directories
mkdir -p data/vector_db data/reports
```

### 2. Learn LangGraph Fundamentals

Start with the learning examples:

```bash
# Example 1: Basic chain
python examples/01_simple_chain.py

# Example 2: Conditional routing
python examples/02_conditional_routing.py

# Example 3: State management
python examples/03_state_management.py
```

See [examples/README.md](examples/README.md) for detailed explanations.

### 3. Run the Research Assistant

```bash
# Basic usage
research "artificial intelligence"

# With options
research "climate change" --max-sources 15 --model llama3.1:8b

# Environment variables (optional)
cp .env.example .env
# Edit .env with your preferences
```

## LangGraph Architecture

### State Flow

```
START
  ↓
query_gen (Generate search queries from topic)
  ↓
search (DuckDuckGo search)
  ↓
scraper (Extract content from URLs)
  ↓
analyzer (Score trustworthiness 0-100)
  ↓
[Conditional Routing]
  ├─→ storage (Store sources >85%) → report
  └─→ report (Generate HTML report)
        ↓
       END
```

### State Schema

```python
class ResearchState(TypedDict):
    topic: str
    search_queries: List[str]
    discovered_urls: Annotated[List[str], add]
    scraped_sources: Annotated[List[Source], add]
    analyzed_sources: Annotated[List[Source], add]
    stored_sources: Annotated[List[Source], add]
    report_html: str
    errors: Annotated[List[str], add]
```

## Implementation Phases

This project is structured for progressive learning:

- **Phase 1**: Setup & LangGraph Fundamentals ✅
  - Project initialization
  - Learning examples

- **Phase 2**: Core Infrastructure
  - Configuration management
  - Data models
  - LLM utilities

- **Phase 3**: Individual Tools
  - DuckDuckGo search
  - Web scraping
  - ChromaDB integration

- **Phase 4**: Agent Implementations
  - Searcher agent
  - Scraper agent
  - Analyzer agent (trustworthiness)
  - Reporter agent

- **Phase 5**: LangGraph Integration
  - State schema
  - Node implementations
  - Graph construction

- **Phase 6**: CLI & Report Generation
  - Command-line interface
  - HTML report templates

- **Phase 7**: Advanced Features (Future)
  - Human-in-the-loop
  - Parallel execution
  - Subgraphs
  - Persistence

See [plan.md](plan.md) for detailed implementation plan.

## Trustworthiness Scoring

Multi-factor scoring approach:

- **Domain Authority (20%)**: .edu, .gov, known sources
- **Content Quality (30%)**: LLM assesses coherence, citations
- **Bias Detection (20%)**: LLM detects propaganda/bias
- **Factual Density (15%)**: Facts vs opinions ratio
- **Recency (10%)**: Publication date
- **Source Transparency (5%)**: Author credentials

**Formula**: Heuristics (40%) + LLM Analysis (60%) = Final Score

Only sources scoring >85% are stored in the vector database.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format
black research_assistant/

# Lint
ruff check research_assistant/

# Type check
mypy research_assistant/
```

## Configuration

Environment variables (`.env`):

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
MAX_SOURCES=10
TRUSTWORTHINESS_THRESHOLD=85
VECTOR_DB_PATH=./data/vector_db
REPORTS_PATH=./data/reports
```

## Future Extensions

- **Multi-model support**: OpenAI, Anthropic, custom models
- **Parallel scraping**: Concurrent URL processing
- **Human-in-the-loop**: Approve sources before storage
- **Subgraphs**: Complex analyzer workflow
- **Persistence**: Save/resume research sessions
- **Streaming**: Real-time result updates
- **Multi-agent debate**: Competing analyzers
- **Web UI**: Streamlit/Gradio interface

## Learning Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://docs.trychroma.com/)
- [examples/README.md](examples/README.md) - Learning examples

## License

MIT License - See LICENSE file for details

## Contributing

This is a learning project! Contributions welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Project Status

🚧 **Phase 1 Complete**: Setup & LangGraph Fundamentals

Next: Implementing Phase 2 (Core Infrastructure)

---

Built with LangGraph for autonomous workflow orchestration.
