"""
Analyzer agent for evaluating source trustworthiness.

Uses LLM-based analysis to score sources based on content quality,
bias detection, factual density, source credibility, and relevance.
"""

import json
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseLLM

from .base import BaseAgent
from ..models.source import Source
from ..utils.prompts import (
    format_trustworthiness_prompt,
    DEFAULT_PREVIEW_LENGTH,
    TEMPERATURE_ANALYTICAL
)
from ..config import settings


class AnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing source trustworthiness.

    Workflow:
    1. For each source, extract content preview and metadata
    2. Use LLM to analyze trustworthiness based on multiple criteria
    3. Parse LLM response to extract score and reasoning
    4. Update source with trustworthiness score and analysis metadata
    5. Return analyzed sources
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        temperature: float = TEMPERATURE_ANALYTICAL,
        verbose: bool = False,
        preview_length: int = DEFAULT_PREVIEW_LENGTH,
        trustworthy_threshold: float = 85.0
    ):
        """
        Initialize analyzer agent.

        Args:
            llm: Language model instance
            temperature: LLM temperature (lower for analytical tasks)
            verbose: Enable verbose logging
            preview_length: Length of content preview for analysis
            trustworthy_threshold: Score threshold for trustworthiness
        """
        super().__init__(llm=llm, temperature=temperature, verbose=verbose)
        self.preview_length = preview_length
        self.trustworthy_threshold = trustworthy_threshold

    def analyze_source(
        self,
        source: Source,
        topic: str
    ) -> Source:
        """
        Analyze a single source for trustworthiness.

        Args:
            source: Source object to analyze
            topic: Research topic for relevance analysis

        Returns:
            Source with updated trustworthiness_score and analysis metadata

        Example:
            >>> agent = AnalyzerAgent(verbose=True)
            >>> analyzed = agent.analyze_source(source, "artificial intelligence")
            >>> print(f"Score: {analyzed.trustworthiness_score}")
        """
        self.log(f"Analyzing: {source.url}")

        try:
            # Get content preview
            content_preview = source.get_content_preview(self.preview_length)

            # Format prompt
            prompt = format_trustworthiness_prompt(
                topic=topic,
                url=source.url,
                title=source.title,
                content_preview=content_preview
            )

            # Get LLM analysis
            response = self.llm.invoke(prompt)

            # Extract content from response
            if hasattr(response, 'content'):
                text = response.content
            else:
                text = str(response)

            # Parse JSON response
            analysis = self._parse_analysis_response(text)

            # Update source with score and analysis
            source.trustworthiness_score = analysis.get('score', 50.0)

            # Add analysis details to metadata
            source.metadata['trustworthiness_analysis'] = {
                'reasoning': analysis.get('reasoning', ''),
                'red_flags': analysis.get('red_flags', []),
                'strengths': analysis.get('strengths', []),
                'analyzed_at': str(source.scraped_at) if source.scraped_at else None
            }

            self.log(f"Score: {source.trustworthiness_score:.1f}/100")
            return source

        except Exception as e:
            self.log(f"Error analyzing {source.url}: {e}", level="ERROR")
            # Return source with default score on error
            source.trustworthiness_score = 50.0
            source.metadata['trustworthiness_analysis'] = {
                'error': str(e),
                'reasoning': 'Analysis failed - assigned default score'
            }
            return source

    def _parse_analysis_response(self, text: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract trustworthiness analysis.

        Args:
            text: LLM response text (expected to be JSON)

        Returns:
            Dictionary with score, reasoning, red_flags, strengths

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Try to find JSON in the response
            # LLM might wrap JSON in markdown code blocks
            text = text.strip()

            # Remove markdown code blocks if present
            if text.startswith('```'):
                # Find the JSON content between ```
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    text = text[start:end]

            # Parse JSON
            analysis = json.loads(text)

            # Validate required fields
            if 'score' not in analysis:
                raise ValueError("Response missing 'score' field")

            # Ensure score is in valid range
            score = float(analysis['score'])
            if not (0 <= score <= 100):
                self.log(f"Score {score} out of range, clamping to 0-100", level="WARNING")
                score = max(0, min(100, score))
                analysis['score'] = score

            # Set defaults for optional fields
            analysis.setdefault('reasoning', 'No reasoning provided')
            analysis.setdefault('red_flags', [])
            analysis.setdefault('strengths', [])

            return analysis

        except json.JSONDecodeError as e:
            self.log(f"Failed to parse JSON: {e}", level="ERROR")
            # Try to extract score from text if JSON parsing fails
            import re
            score_match = re.search(r'score["\s:]+(\d+)', text, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                self.log(f"Extracted score {score} from malformed response", level="WARNING")
                return {
                    'score': score,
                    'reasoning': 'Parsed from malformed response',
                    'red_flags': [],
                    'strengths': []
                }
            raise ValueError(f"Could not parse analysis response: {text[:100]}")

    def analyze_sources(
        self,
        sources: List[Source],
        topic: str
    ) -> List[Source]:
        """
        Analyze multiple sources for trustworthiness.

        Args:
            sources: List of Source objects
            topic: Research topic

        Returns:
            List of analyzed sources with trustworthiness scores

        Example:
            >>> agent = AnalyzerAgent(verbose=True)
            >>> analyzed = agent.analyze_sources(sources, "climate change")
            >>> trustworthy = [s for s in analyzed if s.is_trustworthy()]
        """
        self.log(f"Analyzing {len(sources)} sources")

        analyzed = []
        for i, source in enumerate(sources, 1):
            self.log(f"Progress: {i}/{len(sources)}")
            analyzed_source = self.analyze_source(source, topic)
            analyzed.append(analyzed_source)

        self.log(f"Analysis complete")
        return analyzed

    def run(
        self,
        sources: List[Source],
        topic: str,
        filter_untrustworthy: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute analysis workflow: analyze sources → score → optionally filter.

        Args:
            sources: List of Source objects to analyze
            topic: Research topic
            filter_untrustworthy: Remove sources below threshold (default: False)
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - success: bool
                - sources: List[Source] (analyzed)
                - total_analyzed: int
                - trustworthy_count: int
                - average_score: float
                - filtered_count: int (if filter_untrustworthy=True)
                - error: Optional[str]

        Example:
            >>> agent = AnalyzerAgent(verbose=True)
            >>> result = agent.run(sources=sources, topic="quantum computing")
            >>> print(f"Avg score: {result['average_score']:.1f}")
        """
        try:
            # Validate inputs
            if not sources:
                return self.handle_error(
                    ValueError("No sources provided"),
                    context="AnalyzerAgent.run"
                )

            if not topic:
                return self.handle_error(
                    ValueError("Research topic required"),
                    context="AnalyzerAgent.run"
                )

            # Analyze sources
            analyzed_sources = self.analyze_sources(sources, topic)

            # Calculate statistics
            scores = [s.trustworthiness_score for s in analyzed_sources]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            trustworthy = [
                s for s in analyzed_sources
                if s.trustworthiness_score >= self.trustworthy_threshold
            ]
            trustworthy_count = len(trustworthy)

            # Optionally filter untrustworthy sources
            filtered_count = 0
            if filter_untrustworthy:
                filtered_count = len(analyzed_sources) - len(trustworthy)
                analyzed_sources = trustworthy
                self.log(
                    f"Filtered out {filtered_count} sources below "
                    f"threshold {self.trustworthy_threshold}"
                )

            # Return results
            return self.create_success_result({
                "sources": analyzed_sources,
                "total_analyzed": len(sources),
                "trustworthy_count": trustworthy_count,
                "average_score": avg_score,
                "filtered_count": filtered_count,
                "trustworthy_percentage": (trustworthy_count / len(sources)) * 100
            })

        except Exception as e:
            return self.handle_error(e, context="AnalyzerAgent.run")

    def get_trustworthy_sources(
        self,
        sources: List[Source],
        threshold: Optional[float] = None
    ) -> List[Source]:
        """
        Filter sources by trustworthiness threshold.

        Args:
            sources: List of Source objects
            threshold: Score threshold (uses agent default if None)

        Returns:
            List of trustworthy sources

        Example:
            >>> agent = AnalyzerAgent()
            >>> trustworthy = agent.get_trustworthy_sources(sources, threshold=90.0)
        """
        threshold = threshold or self.trustworthy_threshold
        trustworthy = [s for s in sources if s.trustworthiness_score >= threshold]

        if self.verbose:
            self.log(
                f"Found {len(trustworthy)}/{len(sources)} sources "
                f"above threshold {threshold}"
            )

        return trustworthy

    def get_analysis_statistics(self, sources: List[Source]) -> Dict[str, Any]:
        """
        Get detailed statistics about analyzed sources.

        Args:
            sources: List of analyzed Source objects

        Returns:
            Dictionary with analysis statistics

        Example:
            >>> agent = AnalyzerAgent()
            >>> stats = agent.get_analysis_statistics(sources)
            >>> print(f"Median score: {stats['median_score']}")
        """
        if not sources:
            return {
                "total_sources": 0,
                "average_score": 0.0,
                "median_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "trustworthy_count": 0,
                "trustworthy_percentage": 0.0
            }

        scores = sorted(s.trustworthiness_score for s in sources)
        median_idx = len(scores) // 2
        median_score = scores[median_idx] if scores else 0.0

        trustworthy = self.get_trustworthy_sources(sources)

        return {
            "total_sources": len(sources),
            "average_score": sum(scores) / len(scores),
            "median_score": median_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "trustworthy_count": len(trustworthy),
            "trustworthy_percentage": (len(trustworthy) / len(sources)) * 100,
            "score_distribution": {
                "excellent (90-100)": len([s for s in scores if s >= 90]),
                "good (80-89)": len([s for s in scores if 80 <= s < 90]),
                "fair (70-79)": len([s for s in scores if 70 <= s < 80]),
                "poor (0-69)": len([s for s in scores if s < 70]),
            }
        }
