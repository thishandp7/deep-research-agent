"""
LangGraph workflow construction.

Creates the research assistant StateGraph with all nodes and edges.
"""

from langgraph.graph import StateGraph, END
from typing import Optional

from .state import ResearchState, create_initial_state
from .nodes import (
    query_gen_node,
    search_node,
    scraper_node,
    analyzer_node,
    storage_node,
    report_node,
    should_store_sources,
)


def create_research_graph(verbose: bool = True):
    """
    Create the research assistant workflow graph.

    Workflow:
        START → query_gen → search → scraper → analyzer → [conditional]
                                                               ↓
                                                    storage → report → END
                                                               ↓
                                                          report → END

    Args:
        verbose: Enable verbose logging in nodes

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> graph = create_research_graph()
        >>> result = graph.invoke({"topic": "quantum computing", "max_sources": 10})
        >>> print(result["report_html"])
    """
    # Create graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("search", search_node)
    workflow.add_node("scraper", scraper_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("storage", storage_node)
    workflow.add_node("report", report_node)

    # Set entry point
    workflow.set_entry_point("query_gen")

    # Add sequential edges
    workflow.add_edge("query_gen", "search")
    workflow.add_edge("search", "scraper")
    workflow.add_edge("scraper", "analyzer")

    # Add conditional routing after analyzer
    workflow.add_conditional_edges(
        "analyzer",
        should_store_sources,
        {
            "storage": "storage",  # If trustworthy sources exist
            "report": "report",  # Otherwise skip to report
        },
    )

    # Both paths lead to END
    workflow.add_edge("storage", "report")
    workflow.add_edge("report", END)

    # Compile graph
    return workflow.compile()


def run_research(topic: str, max_sources: int = 10, verbose: bool = True) -> ResearchState:
    """
    Run complete research workflow.

    Convenience function that creates graph and executes research.

    Args:
        topic: Research topic
        max_sources: Maximum number of sources to process
        verbose: Enable verbose logging

    Returns:
        Final ResearchState with all results

    Example:
        >>> result = run_research("artificial general intelligence", max_sources=15)
        >>> print(f"Found {len(result['analyzed_sources'])} sources")
        >>> print(result["report_html"])
    """
    # Create initial state
    initial_state = create_initial_state(topic=topic, max_sources=max_sources)

    # Create and run graph
    graph = create_research_graph(verbose=verbose)
    final_state = graph.invoke(initial_state)

    return final_state


def visualize_graph(output_path: Optional[str] = None) -> None:
    """
    Visualize the research graph.

    Requires graphviz to be installed.

    Args:
        output_path: Optional path to save visualization (PNG)

    Example:
        >>> visualize_graph("research_graph.png")
    """
    try:
        # Create graph
        graph = create_research_graph()

        # Get mermaid representation
        mermaid = graph.get_graph().draw_mermaid()

        if output_path:
            # Save to file
            with open(output_path, "w") as f:
                f.write(mermaid)
            print(f"Graph visualization saved to {output_path}")
        else:
            # Print to console
            print("Research Workflow Graph:")
            print(mermaid)

    except ImportError:
        print("Graphviz not installed. Install with: pip install graphviz")
    except Exception as e:
        print(f"Visualization failed: {e}")


# ============================================================================
# Graph Metadata
# ============================================================================

GRAPH_METADATA = {
    "name": "Research Assistant",
    "version": "0.1.0",
    "description": "LangGraph-based research workflow with trustworthiness analysis",
    "nodes": [
        {
            "name": "query_gen",
            "description": "Generate search queries from topic using LLM",
            "agent": "SearcherAgent",
        },
        {
            "name": "search",
            "description": "Execute DuckDuckGo searches and collect URLs",
            "agent": "SearcherAgent",
        },
        {
            "name": "scraper",
            "description": "Scrape content from discovered URLs",
            "agent": "ScraperAgent",
        },
        {
            "name": "analyzer",
            "description": "Analyze source trustworthiness (0-100 score)",
            "agent": "AnalyzerAgent",
        },
        {
            "name": "storage",
            "description": "Store trustworthy sources (score >= 85) in ChromaDB",
            "tool": "VectorStore",
        },
        {
            "name": "report",
            "description": "Generate professional HTML research report",
            "agent": "ReporterAgent",
        },
    ],
    "edges": {
        "sequential": [
            ("query_gen", "search"),
            ("search", "scraper"),
            ("scraper", "analyzer"),
            ("storage", "report"),
        ],
        "conditional": [
            {
                "from": "analyzer",
                "condition": "should_store_sources",
                "paths": {
                    "storage": "If trustworthy sources exist (score >= 85)",
                    "report": "If no trustworthy sources found",
                },
            }
        ],
    },
}


def print_graph_info():
    """Print information about the research graph."""
    print("=" * 70)
    print(f"Graph: {GRAPH_METADATA['name']} v{GRAPH_METADATA['version']}")
    print(f"Description: {GRAPH_METADATA['description']}")
    print("=" * 70)
    print("\nNodes:")
    for node in GRAPH_METADATA["nodes"]:
        print(f"  - {node['name']}: {node['description']}")
        if "agent" in node:
            print(f"    Uses: {node['agent']}")
        elif "tool" in node:
            print(f"    Uses: {node['tool']}")

    print("\nWorkflow:")
    print("  START → query_gen → search → scraper → analyzer")
    print("                                            ↓")
    print("                                     [conditional]")
    print("                                            ↓")
    print("                          storage (if trustworthy) → report → END")
    print("                                   ↓")
    print("                          report (if none trustworthy) → END")
    print("=" * 70)
