"""
Example 1: Simple LangGraph Chain

This example demonstrates:
- Creating a basic StateGraph
- Defining a state schema with TypedDict
- Adding nodes (functions that transform state)
- Adding edges to connect nodes
- Compiling and invoking the graph

Learning Goals:
- Understand StateGraph basics
- See how state flows through nodes
- Learn node and edge syntax
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


# Step 1: Define the state schema
class SimpleState(TypedDict):
    """State that flows through the graph"""
    topic: str
    queries: list[str]
    results: list[str]


# Step 2: Define node functions
def generate_queries(state: SimpleState) -> dict:
    """
    Node that generates search queries from a topic.

    Each node receives the current state and returns a dict of updates.
    """
    topic = state["topic"]
    print(f"📝 Generating queries for topic: {topic}")

    # Generate 3 simple queries
    queries = [
        f"What is {topic}?",
        f"{topic} explained",
        f"{topic} overview",
    ]

    print(f"   Generated {len(queries)} queries")

    # Return updates to the state
    return {"queries": queries}


def search(state: SimpleState) -> dict:
    """
    Node that simulates searching for each query.

    In a real implementation, this would call a search API.
    """
    queries = state["queries"]
    print(f"🔍 Searching for {len(queries)} queries...")

    # Simulate search results
    results = []
    for query in queries:
        result = f"Result for: {query}"
        results.append(result)

    print(f"   Found {len(results)} results")

    return {"results": results}


def summarize(state: SimpleState) -> dict:
    """
    Node that summarizes the results.
    """
    results = state["results"]
    print(f"📊 Summarizing {len(results)} results...")

    print("\n=== SUMMARY ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")

    return {}  # No state updates needed


# Step 3: Build the graph
def create_simple_graph():
    """Create and compile a simple linear graph"""

    # Initialize the graph with our state schema
    graph = StateGraph(SimpleState)

    # Add nodes to the graph
    graph.add_node("query_gen", generate_queries)
    graph.add_node("search", search)
    graph.add_node("summarize", summarize)

    # Add edges to connect nodes (linear flow)
    graph.add_edge("query_gen", "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)  # END is a special marker

    # Set the entry point (where execution starts)
    graph.set_entry_point("query_gen")

    # Compile the graph into a runnable
    return graph.compile()


# Step 4: Run the graph
if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Simple LangGraph Chain")
    print("=" * 60 + "\n")

    # Create the graph
    app = create_simple_graph()

    # Define initial state
    initial_state: SimpleState = {
        "topic": "artificial intelligence",
        "queries": [],
        "results": [],
    }

    print(f"Starting with topic: '{initial_state['topic']}'\n")

    # Invoke the graph with initial state
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL STATE:")
    print("=" * 60)
    print(f"Topic: {final_state['topic']}")
    print(f"Queries: {final_state['queries']}")
    print(f"Results: {len(final_state['results'])} results")

    print("\n✅ Example completed!")
    print("\nKey Takeaways:")
    print("1. StateGraph manages state flow between nodes")
    print("2. Each node receives state and returns updates")
    print("3. Edges define the execution order")
    print("4. The graph is compiled before execution")
