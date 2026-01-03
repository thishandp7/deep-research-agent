# LangGraph Learning Examples

This directory contains progressive examples to help you learn LangGraph fundamentals before building the full research assistant.

## Learning Path

Work through these examples in order:

### 1. Simple Chain (`01_simple_chain.py`)
**Concepts:** Basic StateGraph, nodes, edges, state flow

Learn how to:
- Define a state schema with TypedDict
- Create nodes (functions that transform state)
- Connect nodes with edges
- Compile and invoke a graph

**Run:**
```bash
python examples/01_simple_chain.py
```

### 2. Conditional Routing (`02_conditional_routing.py`)
**Concepts:** Conditional edges, dynamic routing, branching logic

Learn how to:
- Use conditional_edges for dynamic routing
- Implement routing functions based on state
- Handle multiple execution paths
- Route to different nodes based on conditions

**Run:**
```bash
python examples/02_conditional_routing.py
```

### 3. State Management (`03_state_management.py`)
**Concepts:** State reducers, accumulating lists, complex state

Learn how to:
- Use `Annotated[list[X], add]` for accumulating fields
- Understand replacement vs accumulation
- Manage state across multiple nodes
- Pattern used extensively in the research assistant

**Run:**
```bash
python examples/03_state_management.py
```

## Prerequisites

Make sure you have the dependencies installed:

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Next Steps

After completing these examples, you'll be ready to:
1. Understand the research assistant's graph architecture
2. Build the main StateGraph for the research workflow
3. Implement conditional routing for trustworthiness checks
4. Use state reducers for accumulating sources and errors

## LangGraph Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [StateGraph API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/)
