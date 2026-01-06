"""
Research workflow state schema.

Defines the state that flows through the LangGraph research workflow.
"""

from typing import TypedDict, List, Annotated
from operator import add

from ..models.source import Source


class ResearchState(TypedDict):
    """
    State for the research workflow.

    This state is passed through the LangGraph workflow, with each node
    reading from and updating specific fields.

    Fields with Annotated[List[X], add] accumulate values across nodes
    instead of replacing them.
    """

    # ============================================================================
    # Input Parameters
    # ============================================================================

    topic: str
    """Research topic provided by user"""

    max_sources: int
    """Maximum number of sources to process"""

    # ============================================================================
    # Search Phase
    # ============================================================================

    search_queries: List[str]
    """Generated search queries from topic"""

    discovered_urls: Annotated[List[str], add]
    """URLs discovered from searches (accumulated across queries)"""

    # ============================================================================
    # Scraping Phase
    # ============================================================================

    scraped_sources: Annotated[List[Source], add]
    """Successfully scraped sources with content"""

    failed_urls: Annotated[List[str], add]
    """URLs that failed to scrape"""

    # ============================================================================
    # Analysis Phase
    # ============================================================================

    analyzed_sources: Annotated[List[Source], add]
    """Sources with trustworthiness scores"""

    # ============================================================================
    # Storage Phase
    # ============================================================================

    stored_sources: Annotated[List[Source], add]
    """Sources stored in vector database (trustworthy ones)"""

    rejected_sources: Annotated[List[Source], add]
    """Sources rejected due to low trustworthiness"""

    # ============================================================================
    # Report Phase
    # ============================================================================

    report_html: str
    """Generated HTML report"""

    # ============================================================================
    # Control & Debugging
    # ============================================================================

    current_step: str
    """Current workflow step (for debugging/progress tracking)"""

    errors: Annotated[List[str], add]
    """Accumulated errors during workflow"""


def create_initial_state(topic: str, max_sources: int = 10) -> ResearchState:
    """
    Create initial state for research workflow.

    Args:
        topic: Research topic
        max_sources: Maximum sources to process (default: 10)

    Returns:
        Initial ResearchState with defaults

    Example:
        >>> state = create_initial_state("quantum computing", max_sources=15)
        >>> state["topic"]
        'quantum computing'
    """
    return ResearchState(
        # Input
        topic=topic,
        max_sources=max_sources,
        # Search
        search_queries=[],
        discovered_urls=[],
        # Scraping
        scraped_sources=[],
        failed_urls=[],
        # Analysis
        analyzed_sources=[],
        # Storage
        stored_sources=[],
        rejected_sources=[],
        # Report
        report_html="",
        # Control
        current_step="initialized",
        errors=[],
    )
