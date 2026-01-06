"""
Graph node implementations.

Each node is a function that:
1. Takes ResearchState as input
2. Performs an operation using agents/tools
3. Returns dict with state updates
4. Handles errors gracefully
"""

from typing import Dict, Any
from .state import ResearchState

from ..agents import SearcherAgent, ScraperAgent, AnalyzerAgent, ReporterAgent
from ..tools.vector_store import VectorStore
from ..config import settings


# ============================================================================
# Query Generation Node
# ============================================================================


def query_gen_node(state: ResearchState) -> Dict[str, Any]:
    """
    Generate search queries from research topic.

    Args:
        state: Current research state

    Returns:
        State updates with search_queries
    """
    try:
        topic = state["topic"]
        max_sources = state.get("max_sources", 10)

        # Initialize agent
        agent = SearcherAgent(
            temperature=0.7,  # Higher temperature for creative queries
            verbose=True,
            max_queries=5,
            max_results_per_query=max_sources // 3,  # Distribute across queries
        )

        # Generate queries
        queries = agent.generate_queries(topic)

        return {"search_queries": queries, "current_step": "query_generation_complete"}

    except Exception as e:
        error_msg = f"Query generation failed: {str(e)}"
        return {
            "search_queries": [state["topic"]],  # Fallback to topic itself
            "current_step": "query_generation_failed",
            "errors": [error_msg],
        }


# ============================================================================
# Search Node
# ============================================================================


def search_node(state: ResearchState) -> Dict[str, Any]:
    """
    Execute searches and discover URLs.

    Args:
        state: Current research state

    Returns:
        State updates with discovered_urls
    """
    try:
        queries = state["search_queries"]
        max_sources = state.get("max_sources", 10)

        # Initialize agent
        agent = SearcherAgent(
            verbose=True,
            max_results_per_query=max_sources // len(queries) if queries else max_sources,
        )

        # Execute searches
        urls = agent.search_urls(queries)

        # Limit to max_sources
        urls = urls[:max_sources]

        return {"discovered_urls": urls, "current_step": "search_complete"}

    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        return {"discovered_urls": [], "current_step": "search_failed", "errors": [error_msg]}


# ============================================================================
# Scraper Node
# ============================================================================


def scraper_node(state: ResearchState) -> Dict[str, Any]:
    """
    Scrape content from discovered URLs.

    Args:
        state: Current research state

    Returns:
        State updates with scraped_sources and failed_urls
    """
    try:
        urls = state["discovered_urls"]

        if not urls:
            return {
                "scraped_sources": [],
                "current_step": "scraping_skipped_no_urls",
                "errors": ["No URLs to scrape"],
            }

        # Initialize agent
        agent = ScraperAgent(
            verbose=True,
            timeout=settings.scraper_timeout,
            max_retries=settings.scraper_max_retries,
            skip_errors=True,  # Continue on errors
        )

        # Execute scraping
        result = agent.run(urls=urls, min_success_rate=0.0)

        if not result["success"]:
            raise Exception(result.get("error", "Scraping failed"))

        return {
            "scraped_sources": result["sources"],
            "failed_urls": result["failed_urls"],
            "current_step": "scraping_complete",
        }

    except Exception as e:
        error_msg = f"Scraping failed: {str(e)}"
        return {
            "scraped_sources": [],
            "failed_urls": urls,
            "current_step": "scraping_failed",
            "errors": [error_msg],
        }


# ============================================================================
# Analyzer Node
# ============================================================================


def analyzer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Analyze source trustworthiness.

    Args:
        state: Current research state

    Returns:
        State updates with analyzed_sources
    """
    try:
        sources = state["scraped_sources"]
        topic = state["topic"]

        if not sources:
            return {
                "analyzed_sources": [],
                "current_step": "analysis_skipped_no_sources",
                "errors": ["No sources to analyze"],
            }

        # Initialize agent
        agent = AnalyzerAgent(
            temperature=0.3,  # Lower temperature for analytical tasks
            verbose=True,
            trustworthy_threshold=85.0,
        )

        # Execute analysis
        result = agent.run(
            sources=sources, topic=topic, filter_untrustworthy=False  # Keep all for reporting
        )

        if not result["success"]:
            raise Exception(result.get("error", "Analysis failed"))

        return {"analyzed_sources": result["sources"], "current_step": "analysis_complete"}

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        # Return sources with default scores on error
        for source in sources:
            if source.trustworthiness_score == 0.0:
                source.trustworthiness_score = 50.0

        return {
            "analyzed_sources": sources,
            "current_step": "analysis_failed",
            "errors": [error_msg],
        }


# ============================================================================
# Storage Node
# ============================================================================


def storage_node(state: ResearchState) -> Dict[str, Any]:
    """
    Store trustworthy sources in vector database.

    Args:
        state: Current research state

    Returns:
        State updates with stored_sources and rejected_sources
    """
    try:
        sources = state["analyzed_sources"]

        if not sources:
            return {
                "stored_sources": [],
                "rejected_sources": [],
                "current_step": "storage_skipped_no_sources",
            }

        # Separate trustworthy from rejected
        trustworthy = [s for s in sources if s.trustworthiness_score >= 85.0]
        rejected = [s for s in sources if s.trustworthiness_score < 85.0]

        if not trustworthy:
            return {
                "stored_sources": [],
                "rejected_sources": rejected,
                "current_step": "storage_skipped_no_trustworthy_sources",
            }

        # Initialize vector store
        vector_store = VectorStore(
            collection_name=f"research_{state['topic'][:30]}",
            persist_directory=str(settings.vector_db_path),
        )

        # Store sources
        vector_store.add_sources(trustworthy)

        return {
            "stored_sources": trustworthy,
            "rejected_sources": rejected,
            "current_step": "storage_complete",
        }

    except Exception as e:
        error_msg = f"Storage failed: {str(e)}"
        return {
            "stored_sources": [],
            "rejected_sources": sources,
            "current_step": "storage_failed",
            "errors": [error_msg],
        }


# ============================================================================
# Report Node
# ============================================================================


def report_node(state: ResearchState) -> Dict[str, Any]:
    """
    Generate HTML research report.

    Args:
        state: Current research state

    Returns:
        State updates with report_html
    """
    try:
        topic = state["topic"]
        sources = state["analyzed_sources"]

        if not sources:
            # Generate error report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Research Report: {topic}</title></head>
            <body>
                <h1>Research Report: {topic}</h1>
                <p>No sources were successfully analyzed.</p>
                <p>Errors: {', '.join(state.get('errors', ['None']))}</p>
            </body>
            </html>
            """
            return {"report_html": html, "current_step": "report_generated_with_errors"}

        # Initialize agent
        agent = ReporterAgent(
            temperature=0.7,  # Balanced for report generation
            verbose=True,
            include_full_content=False,  # Just previews
            max_sources_in_report=None,  # Include all
        )

        # Generate report
        result = agent.run(
            topic=topic, sources=sources, output_path=None  # Don't save to file here
        )

        if not result["success"]:
            raise Exception(result.get("error", "Report generation failed"))

        return {"report_html": result["html_report"], "current_step": "report_complete"}

    except Exception as e:
        error_msg = f"Report generation failed: {str(e)}"
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error Report: {state['topic']}</title></head>
        <body>
            <h1>Report Generation Error</h1>
            <p>{error_msg}</p>
        </body>
        </html>
        """
        return {"report_html": html, "current_step": "report_failed", "errors": [error_msg]}


# ============================================================================
# Conditional Routing Functions
# ============================================================================


def should_store_sources(state: ResearchState) -> str:
    """
    Determine whether to store sources or skip to report.

    Args:
        state: Current research state

    Returns:
        "storage" if trustworthy sources exist, else "report"
    """
    sources = state.get("analyzed_sources", [])
    trustworthy = [s for s in sources if s.trustworthiness_score >= 85.0]

    if trustworthy:
        return "storage"
    else:
        return "report"
