# Deep Research Assistant - Implementation Plan

## Project Overview
Build a LangGraph-based research assistant that searches the web, evaluates source trustworthiness, and generates comprehensive research reports. This project will demonstrate LangGraph fundamentals through advanced patterns.

## Tech Stack
- **LangGraph**: Workflow orchestration
- **Ollama**: Open-source LLM (llama3.2:3b for quick iteration, llama3.1:8b for quality)
- **ChromaDB**: Vector database for storing vetted sources
- **DuckDuckGo**: Free web search (no API key required)
- **HTML**: Report generation format
- **uv**: Python package management

## Project Structure
```
ResearchAssistant/
├── pyproject.toml              # Dependencies (uv)
├── README.md                   # Documentation
├── .gitignore
├── .env.example
│
├── research_assistant/         # Main package
│   ├── graph/                  # LangGraph orchestration
│   │   ├── state.py            # State schema (ResearchState)
│   │   ├── graph.py            # Graph construction
│   │   └── nodes.py            # Node implementations
│   │
│   ├── agents/                 # Specialized agents
│   │   ├── base.py             # Base agent class
│   │   ├── searcher.py         # Web search
│   │   ├── scraper.py          # Content scraping
│   │   ├── analyzer.py         # Trustworthiness scoring
│   │   └── reporter.py         # Report generation
│   │
│   ├── tools/                  # Agent tools
│   │   ├── search.py           # DuckDuckGo integration
│   │   ├── scraper.py          # Web scraping utilities
│   │   └── vector_store.py     # ChromaDB operations
│   │
│   ├── models/                 # Data models
│   │   └── source.py           # Source data model
│   │
│   ├── utils/                  # Utilities
│   │   ├── llm.py              # LLM initialization
│   │   ├── prompts.py          # Prompt templates
│   │   └── html_generator.py  # HTML report generation
│   │
│   ├── config.py               # Configuration
│   └── main.py                 # CLI entry point
│
├── examples/                   # Learning progression
│   ├── 01_simple_chain.py      # Basic LangGraph chain
│   ├── 02_conditional_routing.py
│   ├── 03_state_management.py
│   └── 04_full_research.py
│
├── data/                       # Runtime data
│   ├── vector_db/              # ChromaDB storage
│   └── reports/                # Generated reports
│
└── tests/                      # Tests
```

## LangGraph Architecture

### State Schema (ResearchState)
```python
class ResearchState(TypedDict):
    # Input
    topic: str
    max_sources: int

    # Search phase
    search_queries: List[str]
    discovered_urls: Annotated[List[str], add]

    # Scraping phase
    scraped_sources: Annotated[List[Source], add]
    failed_urls: Annotated[List[str], add]

    # Analysis phase
    analyzed_sources: Annotated[List[Source], add]

    # Storage phase
    stored_sources: Annotated[List[Source], add]
    rejected_sources: Annotated[List[Source], add]

    # Report
    report_html: str

    # Control
    current_step: str
    errors: Annotated[List[str], add]
```

### Graph Flow
```
START → query_gen → search → scraper → analyzer → [conditional] → storage → report → END
                                                                 ↓
                                                              report → END
```

**Nodes:**
1. **query_gen**: Generate 3-5 search queries from topic using LLM
2. **search**: Execute DuckDuckGo searches, collect URLs
3. **scraper**: Scrape content from URLs
4. **analyzer**: Score trustworthiness (0-100)
5. **storage**: Store sources with score > 85 in ChromaDB
6. **report**: Generate HTML report

**Conditional Routing:**
- After analyzer: If trustworthy sources exist (score > 85) → storage, else → report

## Implementation Phases

### Phase 1: Setup & LangGraph Fundamentals (Day 1-2)
**Goal:** Initialize project and learn LangGraph basics

**Tasks:**
1. Initialize project with uv
   - Create pyproject.toml with dependencies
   - Setup virtual environment
   - Install Ollama and pull models (llama3.2:3b, llama3.1:8b)

2. Create learning examples:
   - `examples/01_simple_chain.py` - Linear 2-3 node graph
   - `examples/02_conditional_routing.py` - Conditional edges
   - `examples/03_state_management.py` - State with reducers

**Learning Outcomes:**
- StateGraph creation
- Adding nodes/edges
- State schema design
- Conditional routing
- Graph visualization

### Phase 2: Core Infrastructure (Day 3)
**Goal:** Build foundational components

**Files to Create:**
- `research_assistant/__init__.py`
- `research_assistant/config.py` - Configuration management
- `research_assistant/models/source.py` - Source data model (Pydantic)
- `research_assistant/utils/llm.py` - LLM initialization with model switching
- `research_assistant/utils/prompts.py` - Prompt templates

**Key Design:**
- LLM abstraction to support model switching
- Pydantic models for data validation

### Phase 3: Individual Tools (Day 4-5)
**Goal:** Build and test tools independently

**Files to Create:**
1. `research_assistant/tools/search.py`
   - DuckDuckGo integration using `duckduckgo-search`
   - Return top N URLs per query

2. `research_assistant/tools/scraper.py`
   - BeautifulSoup/newspaper3k for content extraction
   - Error handling for failed scrapes

3. `research_assistant/tools/vector_store.py`
   - ChromaDB setup and configuration
   - Add/query operations
   - Automatic embedding generation

**Testing:** Test each tool in isolation with mock data

### Phase 4: Agent Implementations (Day 6-7)
**Goal:** Build specialized agents

**Files to Create:**
1. `research_assistant/agents/base.py` - Base agent class
2. `research_assistant/agents/searcher.py` - Query generation + search
3. `research_assistant/agents/scraper.py` - Content scraping
4. `research_assistant/agents/analyzer.py` - Trustworthiness scoring
5. `research_assistant/agents/reporter.py` - HTML report generation

**Critical: Trustworthiness Scoring (analyzer.py)**
Multi-factor approach:
- Domain authority (20%): .edu, .gov, known domains
- Content quality (30%): LLM assesses coherence, citations
- Bias detection (20%): LLM detects propaganda/bias
- Factual density (15%): Facts vs opinions ratio
- Recency (10%): Publication date
- Source transparency (5%): Author credentials

Hybrid scoring: Heuristics (40%) + LLM analysis (60%) = final score

### Phase 5: LangGraph Integration (Day 8-9)
**Goal:** Build the complete research graph

**Files to Create:**
1. `research_assistant/graph/state.py` - ResearchState schema
2. `research_assistant/graph/nodes.py` - Node implementations
3. `research_assistant/graph/graph.py` - Graph construction

**Node Implementation Pattern:**
Each node:
- Takes ResearchState as input
- Returns dict with state updates
- Handles errors gracefully
- Updates current_step field

**Conditional Routing:**
```python
def should_store_sources(state: ResearchState) -> str:
    trustworthy = [s for s in state["analyzed_sources"]
                   if s.trustworthiness_score > 85]
    return "storage" if trustworthy else "report"
```

### Phase 6: CLI & Report Generation (Day 10)
**Goal:** Complete user interface

**Files to Create:**
1. `research_assistant/main.py`
   - CLI using argparse
   - Rich console output with progress bars
   - Invoke graph and save report

2. `research_assistant/utils/html_generator.py`
   - Jinja2 templates
   - Professional HTML report with:
     - Executive summary
     - Sources list with trust scores
     - Key findings
     - Full content

**CLI Usage:**
```bash
research "artificial general intelligence" --max-sources 15 --model llama3.1:8b
```

### Phase 7: Advanced Features (Future)
**Goal:** Master advanced LangGraph patterns

**Possible Extensions:**
- Human-in-the-loop: Approve sources before storage
- Parallel execution: Scrape URLs concurrently
- Subgraphs: Break analyzer into sub-workflow
- Persistence: Save/resume research sessions
- Streaming: Stream results as discovered
- Multi-agent debate: Competing analyzer agents

## Key Dependencies

```toml
[project]
dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.20",
    "ollama>=0.1.0",
    "duckduckgo-search>=4.0.0",
    "beautifulsoup4>=4.12.0",
    "newspaper3k>=0.2.8",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "jinja2>=3.1.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
]
```

## Critical Files (Priority Order)

1. **examples/01_simple_chain.py** - Start here for LangGraph learning
2. **research_assistant/graph/state.py** - Core state schema
3. **research_assistant/graph/graph.py** - Graph construction and orchestration
4. **research_assistant/graph/nodes.py** - Node implementations
5. **research_assistant/agents/analyzer.py** - Trustworthiness scoring logic
6. **research_assistant/tools/vector_store.py** - ChromaDB integration
7. **research_assistant/main.py** - CLI entry point

## Design Decisions

### Why ChromaDB over FAISS?
- Higher-level API (easier learning curve)
- Built-in metadata filtering
- Automatic persistence
- Auto-embedding generation
- Can swap to FAISS later via abstraction

### Why Single Graph vs Multi-Agent?
- Clearer control flow for learning
- Easier debugging
- Explicit state transitions
- Better demonstrates LangGraph patterns
- Can add agent-based nodes later

### LLM Model Strategy
- Start: llama3.2:3b (fast iteration)
- Production: llama3.1:8b (better quality)
- Future: Abstract interface for OpenAI/Anthropic/custom models

### State Management Strategy
- Use `Annotated[List[X], add]` for accumulating lists
- Separate fields for each pipeline phase
- Include error tracking in state
- Track current_step for debugging

## Setup Commands

```bash
# 1. Initialize project
cd /Users/thishandp7/Code/ResearchAssistant
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -e ".[dev]"

# 3. Setup Ollama
ollama pull llama3.2:3b
ollama pull llama3.1:8b

# 4. Create directories
mkdir -p data/vector_db data/reports

# 5. Initialize git (optional)
git init
```

## Testing Strategy

1. **Unit tests**: Test each agent/tool independently
2. **Integration tests**: Test graph workflow end-to-end
3. **Example scripts**: Run examples to validate LangGraph concepts
4. **Manual testing**: Run CLI with various research topics

## Success Criteria

**Functional:**
- ✅ Accepts research topic, generates queries
- ✅ Searches web and discovers sources
- ✅ Scrapes content from URLs
- ✅ Scores trustworthiness accurately
- ✅ Stores high-quality sources (>85%) in ChromaDB
- ✅ Generates professional HTML reports

**Learning:**
- ✅ Understanding of LangGraph StateGraph
- ✅ Mastery of state schema and reducers
- ✅ Conditional routing implementation
- ✅ Node design patterns
- ✅ Error handling in graphs
- ✅ Ready for advanced patterns (parallel, subgraphs, human-in-loop)

## Next Steps After Implementation

1. Add more sophisticated trustworthiness scoring
2. Implement parallel URL scraping
3. Add human-in-the-loop approval
4. Create subgraph for complex analysis
5. Add persistence for resumable research
6. Experiment with multi-agent debate patterns
7. Build web UI (Streamlit/Gradio)
