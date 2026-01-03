"""
Example 3: Advanced State Management with Reducers

This example demonstrates:
- Using Annotated types with operator.add for accumulating lists
- State reducers that combine updates from multiple nodes
- Managing complex state transformations
- How state updates are merged vs replaced

Learning Goals:
- Understand state reducers (Annotated with operator.add)
- See how lists accumulate across node executions
- Learn the difference between replacement and accumulation
- Practice patterns used in the research assistant
"""

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END


# Step 1: Define state with reducers
class ResearchState(TypedDict):
    """
    State with both regular fields and accumulating fields.

    Regular fields: Last value wins (replacement)
    Annotated[List[X], add]: Values accumulate (concatenation)
    """

    # Regular field - gets replaced on each update
    current_phase: str

    # Accumulating fields - values get concatenated using operator.add
    discovered_urls: Annotated[list[str], add]
    processed_urls: Annotated[list[str], add]
    errors: Annotated[list[str], add]

    # Regular field for counting
    total_processed: int


# Step 2: Define nodes that demonstrate different update patterns
def discover_urls_batch1(state: ResearchState) -> dict:
    """
    First batch of URL discovery.
    Demonstrates accumulating list updates.
    """
    print("🔍 Discovering URLs - Batch 1")

    new_urls = [
        "https://example.com/article1",
        "https://example.com/article2",
    ]

    print(f"   Found {len(new_urls)} URLs")

    # Because discovered_urls uses Annotated[list, add],
    # these URLs will be ADDED to existing ones, not replace them
    return {
        "discovered_urls": new_urls,
        "current_phase": "discovery_batch1",
    }


def discover_urls_batch2(state: ResearchState) -> dict:
    """
    Second batch of URL discovery.
    Shows how multiple nodes can accumulate to the same field.
    """
    print("\n🔍 Discovering URLs - Batch 2")

    new_urls = [
        "https://example.com/article3",
        "https://example.com/article4",
        "https://example.com/article5",
    ]

    print(f"   Found {len(new_urls)} URLs")

    # These URLs will be ADDED to the previous batch
    return {
        "discovered_urls": new_urls,
        "current_phase": "discovery_batch2",
    }


def process_urls(state: ResearchState) -> dict:
    """
    Process discovered URLs.
    Demonstrates accessing accumulated state and adding to different lists.
    """
    print("\n⚙️  Processing URLs")

    urls = state["discovered_urls"]
    print(f"   Total URLs to process: {len(urls)}")

    processed = []
    errors = []

    # Simulate processing each URL
    for i, url in enumerate(urls, 1):
        print(f"   Processing {i}/{len(urls)}: {url}")

        # Simulate: even-indexed URLs fail
        if i % 2 == 0:
            errors.append(f"Failed to process: {url}")
        else:
            processed.append(url)

    print(f"   ✓ Processed: {len(processed)}")
    print(f"   ✗ Errors: {len(errors)}")

    # Both processed_urls and errors will accumulate
    return {
        "processed_urls": processed,
        "errors": errors,
        "total_processed": len(processed),
        "current_phase": "processing",
    }


def summarize_results(state: ResearchState) -> dict:
    """
    Final summary node.
    Shows how to read accumulated state.
    """
    print("\n📊 Generating Summary")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTotal URLs discovered: {len(state['discovered_urls'])}")
    print(f"Successfully processed: {len(state['processed_urls'])}")
    print(f"Errors encountered: {len(state['errors'])}")

    if state["processed_urls"]:
        print("\nSuccessfully Processed:")
        for url in state["processed_urls"]:
            print(f"  ✓ {url}")

    if state["errors"]:
        print("\nErrors:")
        for error in state["errors"]:
            print(f"  ✗ {error}")

    return {"current_phase": "complete"}


# Step 3: Build the graph
def create_state_management_graph():
    """Create a graph demonstrating state management"""

    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("discover1", discover_urls_batch1)
    graph.add_node("discover2", discover_urls_batch2)
    graph.add_node("process", process_urls)
    graph.add_node("summarize", summarize_results)

    # Linear flow
    graph.set_entry_point("discover1")
    graph.add_edge("discover1", "discover2")
    graph.add_edge("discover2", "process")
    graph.add_edge("process", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


# Step 4: Run the example
if __name__ == "__main__":
    print("=" * 60)
    print("Example 3: Advanced State Management")
    print("=" * 60 + "\n")

    app = create_state_management_graph()

    # Initial state with empty accumulating lists
    initial_state: ResearchState = {
        "current_phase": "start",
        "discovered_urls": [],  # Will accumulate
        "processed_urls": [],  # Will accumulate
        "errors": [],  # Will accumulate
        "total_processed": 0,
    }

    print("Initial state:")
    print(f"  - discovered_urls: {initial_state['discovered_urls']}")
    print(f"  - processed_urls: {initial_state['processed_urls']}")
    print(f"  - errors: {initial_state['errors']}")
    print()

    # Execute the graph
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"Current phase: {final_state['current_phase']}")
    print(f"Discovered URLs: {len(final_state['discovered_urls'])} total")
    print(f"Processed URLs: {len(final_state['processed_urls'])} total")
    print(f"Errors: {len(final_state['errors'])} total")
    print(f"Total processed count: {final_state['total_processed']}")

    print("\n✅ Example completed!")
    print("\nKey Takeaways:")
    print("1. Annotated[list[X], add] accumulates values across nodes")
    print("2. Regular fields get replaced with latest value")
    print("3. Multiple nodes can contribute to the same accumulating field")
    print("4. This pattern is perfect for collecting results from multiple operations")
    print("\nReal-world use case:")
    print("- Our research assistant uses this to accumulate:")
    print("  • discovered_urls from multiple search queries")
    print("  • scraped_sources from parallel scraping")
    print("  • analyzed_sources from trustworthiness checks")
    print("  • errors from various failure points")
