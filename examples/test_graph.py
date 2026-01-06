"""
Example: Testing the Research Graph

This script demonstrates how to use the research graph workflow.
It's a minimal example to verify the graph is working correctly.
"""

from research_assistant.graph import create_research_graph
from research_assistant.graph.state import create_initial_state
from research_assistant.graph.graph import print_graph_info


def main():
    """Run a simple research workflow."""
    # Print graph information
    print_graph_info()

    print("\n" + "=" * 70)
    print("RUNNING RESEARCH WORKFLOW")
    print("=" * 70 + "\n")

    # Create initial state
    topic = "quantum computing basics"
    max_sources = 5

    print(f"Topic: {topic}")
    print(f"Max sources: {max_sources}\n")

    initial_state = create_initial_state(topic=topic, max_sources=max_sources)

    # Create graph
    print("Creating graph...")
    graph = create_research_graph(verbose=True)
    print("✓ Graph created\n")

    # Execute workflow
    print("Executing workflow...\n")
    print("-" * 70)

    try:
        final_state = graph.invoke(initial_state)

        print("-" * 70)
        print("\n✓ Workflow completed successfully!\n")

        # Print results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nCurrent Step: {final_state['current_step']}")
        print(f"Errors: {len(final_state.get('errors', []))}")

        print(f"\nSearch Queries Generated: {len(final_state['search_queries'])}")
        for i, query in enumerate(final_state["search_queries"], 1):
            print(f"  {i}. {query}")

        print(f"\nURLs Discovered: {len(final_state['discovered_urls'])}")
        print(f"Sources Scraped: {len(final_state['scraped_sources'])}")
        print(f"Failed URLs: {len(final_state['failed_urls'])}")

        print(f"\nSources Analyzed: {len(final_state['analyzed_sources'])}")
        if final_state["analyzed_sources"]:
            scores = [s.trustworthiness_score for s in final_state["analyzed_sources"]]
            print(f"  Average Score: {sum(scores)/len(scores):.1f}")
            print(f"  Score Range: {min(scores):.1f} - {max(scores):.1f}")

        print(f"\nTrustworthy Sources Stored: {len(final_state['stored_sources'])}")
        print(f"Rejected Sources: {len(final_state['rejected_sources'])}")

        print(f"\nReport Generated: {'Yes' if final_state['report_html'] else 'No'}")
        if final_state["report_html"]:
            report_length = len(final_state["report_html"])
            print(f"  Report Size: {report_length:,} characters")

            # Optionally save report
            output_path = f"data/reports/{topic.replace(' ', '_')}_report.html"
            import os

            os.makedirs("data/reports", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_state["report_html"])
            print(f"  Saved to: {output_path}")

        if final_state.get("errors"):
            print("\nErrors encountered:")
            for error in final_state["errors"]:
                print(f"  - {error}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
