"""
Searcher agent for generating search queries and discovering sources.

Combines LLM-based query generation with web search to find relevant URLs.
"""

from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseLLM

from .base import BaseAgent
from ..tools.search import search_multiple_queries, search_duckduckgo
from ..utils.prompts import QUERY_GENERATION_TEMPLATE


class SearcherAgent(BaseAgent):
    """
    Agent responsible for generating search queries and discovering URLs.

    Workflow:
    1. Generate 3-5 search queries from research topic using LLM
    2. Execute searches via DuckDuckGo
    3. Deduplicate and return URLs
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        temperature: float = 0.7,
        verbose: bool = False,
        max_queries: int = 5,
        max_results_per_query: int = 10,
    ):
        """
        Initialize searcher agent.

        Args:
            llm: Language model instance
            temperature: LLM temperature (higher for creative queries)
            verbose: Enable verbose logging
            max_queries: Maximum number of queries to generate
            max_results_per_query: Max results per query
        """
        super().__init__(llm=llm, temperature=temperature, verbose=verbose)
        self.max_queries = max_queries
        self.max_results_per_query = max_results_per_query

    def generate_queries(self, topic: str) -> List[str]:
        """
        Generate search queries from research topic using LLM.

        Args:
            topic: Research topic

        Returns:
            List of search query strings

        Example:
            >>> agent = SearcherAgent()
            >>> queries = agent.generate_queries("artificial intelligence ethics")
            >>> print(queries)
            ['AI ethics principles', 'ethical concerns in AI', ...]
        """
        self.log(f"Generating queries for topic: {topic}")

        try:
            # Format prompt
            prompt = QUERY_GENERATION_TEMPLATE.format(topic=topic, max_queries=self.max_queries)

            # Get LLM response
            response = self.llm.invoke(prompt)

            # Extract content from response
            if hasattr(response, "content"):
                text = response.content
            else:
                text = str(response)

            # Parse queries from response (assumes one per line)
            queries = [
                line.strip().lstrip("123456789.-) ")
                for line in text.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            # Limit to max_queries
            queries = queries[: self.max_queries]

            # Fallback: use topic itself if no queries generated
            if not queries:
                self.log("No queries generated, using topic as fallback", level="WARNING")
                queries = [topic]

            self.log(f"Generated {len(queries)} queries")
            return queries

        except Exception as e:
            self.log(f"Error generating queries: {e}", level="ERROR")
            # Fallback to topic
            return [topic]

    def search_urls(self, queries: List[str]) -> List[str]:
        """
        Execute searches and collect URLs.

        Args:
            queries: List of search queries

        Returns:
            Deduplicated list of URLs

        Example:
            >>> agent = SearcherAgent()
            >>> urls = agent.search_urls(['AI ethics', 'machine learning'])
            >>> print(f"Found {len(urls)} URLs")
        """
        self.log(f"Searching {len(queries)} queries")

        try:
            urls = search_multiple_queries(
                queries, max_results_per_query=self.max_results_per_query, deduplicate=True
            )

            self.log(f"Discovered {len(urls)} unique URLs")
            return urls

        except Exception as e:
            self.log(f"Error during search: {e}", level="ERROR")
            return []

    def run(self, topic: str, queries: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute search workflow: generate queries → search → collect URLs.

        Args:
            topic: Research topic
            queries: Pre-generated queries (optional, will generate if None)
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - success: bool
                - queries: List[str]
                - urls: List[str]
                - error: Optional[str]

        Example:
            >>> agent = SearcherAgent(verbose=True)
            >>> result = agent.run(topic="quantum computing")
            >>> print(f"Found {len(result['urls'])} URLs")
        """
        try:
            # Generate queries if not provided
            if queries is None:
                queries = self.generate_queries(topic)
            else:
                self.log(f"Using {len(queries)} provided queries")

            # Execute searches
            urls = self.search_urls(queries)

            # Return results
            return self.create_success_result(
                {"queries": queries, "urls": urls, "url_count": len(urls)}
            )

        except Exception as e:
            return self.handle_error(e, context="SearcherAgent.run")

    def run_simple_search(self, query: str, max_results: int = 10) -> List[str]:
        """
        Simple single-query search (utility method).

        Args:
            query: Single search query
            max_results: Maximum results to return

        Returns:
            List of URLs

        Example:
            >>> agent = SearcherAgent()
            >>> urls = agent.run_simple_search("python programming")
        """
        try:
            results = search_duckduckgo(query, max_results=max_results)
            return [r["href"] for r in results]
        except Exception as e:
            self.log(f"Simple search failed: {e}", level="ERROR")
            return []
