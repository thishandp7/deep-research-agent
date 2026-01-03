"""
Example 2: Conditional Routing in LangGraph

This example demonstrates:
- Conditional edges based on state
- Dynamic routing between different nodes
- Using routing functions to control flow
- Handling different execution paths

Learning Goals:
- Understand conditional_edges
- Implement routing logic
- See how graphs can branch based on state
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END


# Step 1: Define the state schema
class RoutingState(TypedDict):
    """State for conditional routing example"""
    content: str
    quality_score: int
    path_taken: str
    is_approved: bool


# Step 2: Define node functions
def analyze_content(state: RoutingState) -> dict:
    """
    Analyze content and assign a quality score.
    This simulates content quality analysis.
    """
    content = state["content"]
    print(f"🔍 Analyzing content: '{content[:50]}...'")

    # Simple heuristic: longer content = higher quality
    quality_score = min(100, len(content) * 2)

    print(f"   Quality score: {quality_score}/100")

    return {
        "quality_score": quality_score,
        "path_taken": "analysis",
    }


def approve_content(state: RoutingState) -> dict:
    """
    Handle high-quality content.
    """
    print(f"✅ Content APPROVED (score: {state['quality_score']})")
    print(f"   Content meets quality standards!")

    return {
        "is_approved": True,
        "path_taken": state["path_taken"] + " → approval",
    }


def reject_content(state: RoutingState) -> dict:
    """
    Handle low-quality content.
    """
    print(f"❌ Content REJECTED (score: {state['quality_score']})")
    print(f"   Content needs improvement.")

    return {
        "is_approved": False,
        "path_taken": state["path_taken"] + " → rejection",
    }


def needs_review(state: RoutingState) -> dict:
    """
    Handle medium-quality content that needs human review.
    """
    print(f"⚠️  Content needs HUMAN REVIEW (score: {state['quality_score']})")
    print(f"   Quality is borderline.")

    return {
        "is_approved": False,
        "path_taken": state["path_taken"] + " → review",
    }


# Step 3: Define routing function
def route_based_on_quality(state: RoutingState) -> Literal["approve", "reject", "review"]:
    """
    Routing function that determines next node based on quality score.

    This is the key to conditional edges:
    - Takes state as input
    - Returns the name of the next node to execute
    """
    score = state["quality_score"]

    print(f"\n🔀 Routing decision for score {score}:")

    if score >= 80:
        print(f"   → Route to APPROVE (score >= 80)")
        return "approve"
    elif score >= 50:
        print(f"   → Route to REVIEW (50 <= score < 80)")
        return "review"
    else:
        print(f"   → Route to REJECT (score < 50)")
        return "reject"


# Step 4: Build the graph with conditional routing
def create_routing_graph():
    """Create a graph with conditional edges"""

    graph = StateGraph(RoutingState)

    # Add all nodes
    graph.add_node("analyze", analyze_content)
    graph.add_node("approve", approve_content)
    graph.add_node("reject", reject_content)
    graph.add_node("review", needs_review)

    # Set entry point
    graph.set_entry_point("analyze")

    # Add conditional edges from analyze node
    # The routing function determines which node to go to next
    graph.add_conditional_edges(
        "analyze",  # Source node
        route_based_on_quality,  # Routing function
        {
            # Map of return values to target nodes
            "approve": "approve",
            "reject": "reject",
            "review": "review",
        },
    )

    # All paths converge to END
    graph.add_edge("approve", END)
    graph.add_edge("reject", END)
    graph.add_edge("review", END)

    return graph.compile()


# Step 5: Run multiple examples
if __name__ == "__main__":
    print("=" * 60)
    print("Example 2: Conditional Routing")
    print("=" * 60 + "\n")

    app = create_routing_graph()

    # Test case 1: High quality (should approve)
    print("\n" + "─" * 60)
    print("TEST CASE 1: High Quality Content")
    print("─" * 60)

    state1: RoutingState = {
        "content": "This is a very long and comprehensive article about artificial intelligence " * 5,
        "quality_score": 0,
        "path_taken": "",
        "is_approved": False,
    }

    result1 = app.invoke(state1)
    print(f"\nPath: {result1['path_taken']}")
    print(f"Approved: {result1['is_approved']}")

    # Test case 2: Low quality (should reject)
    print("\n" + "─" * 60)
    print("TEST CASE 2: Low Quality Content")
    print("─" * 60)

    state2: RoutingState = {
        "content": "Short text",
        "quality_score": 0,
        "path_taken": "",
        "is_approved": False,
    }

    result2 = app.invoke(state2)
    print(f"\nPath: {result2['path_taken']}")
    print(f"Approved: {result2['is_approved']}")

    # Test case 3: Medium quality (should review)
    print("\n" + "─" * 60)
    print("TEST CASE 3: Medium Quality Content")
    print("─" * 60)

    state3: RoutingState = {
        "content": "This is a moderately detailed article about the topic with some good points.",
        "quality_score": 0,
        "path_taken": "",
        "is_approved": False,
    }

    result3 = app.invoke(state3)
    print(f"\nPath: {result3['path_taken']}")
    print(f"Approved: {result3['is_approved']}")

    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("\nKey Takeaways:")
    print("1. Conditional edges enable dynamic routing")
    print("2. Routing functions decide next node based on state")
    print("3. Multiple paths can converge back to END")
    print("4. This pattern is essential for complex workflows")
