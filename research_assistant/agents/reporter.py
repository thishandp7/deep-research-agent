"""
Reporter agent for generating HTML research reports.

Creates comprehensive, well-formatted HTML reports from analyzed sources
with executive summaries, key findings, and source listings.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.language_models import BaseLLM

from .base import BaseAgent
from ..models.source import Source
from ..utils.prompts import (
    format_report_summary_prompt,
    format_key_findings_prompt,
    TEMPERATURE_BALANCED,
)


class ReporterAgent(BaseAgent):
    """
    Agent responsible for generating research reports.

    Workflow:
    1. Receive analyzed sources
    2. Generate executive summary using LLM
    3. Extract key findings using LLM
    4. Format sources with metadata
    5. Generate HTML report with styling
    6. Return report content
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        temperature: float = TEMPERATURE_BALANCED,
        verbose: bool = False,
        include_full_content: bool = False,
        max_sources_in_report: Optional[int] = None,
    ):
        """
        Initialize reporter agent.

        Args:
            llm: Language model instance
            temperature: LLM temperature for report generation
            verbose: Enable verbose logging
            include_full_content: Include full source content in report
            max_sources_in_report: Maximum sources to include (None = all)
        """
        super().__init__(llm=llm, temperature=temperature, verbose=verbose)
        self.include_full_content = include_full_content
        self.max_sources_in_report = max_sources_in_report

    def generate_executive_summary(self, topic: str, sources: List[Source]) -> str:
        """
        Generate executive summary from sources using LLM.

        Args:
            topic: Research topic
            sources: List of analyzed Source objects

        Returns:
            Executive summary text

        Example:
            >>> agent = ReporterAgent(verbose=True)
            >>> summary = agent.generate_executive_summary("AI ethics", sources)
        """
        self.log(f"Generating executive summary for {len(sources)} sources")

        try:
            # Create sources summary for prompt
            sources_summary = self._format_sources_for_prompt(sources)

            # Format prompt
            prompt = format_report_summary_prompt(
                topic=topic, sources_summary=sources_summary, num_sources=len(sources)
            )

            # Get LLM response
            response = self.llm.invoke(prompt)

            # Extract content
            if hasattr(response, "content"):
                summary = response.content
            else:
                summary = str(response)

            self.log("Executive summary generated")
            return summary.strip()

        except Exception as e:
            self.log(f"Error generating summary: {e}", level="ERROR")
            return f"Error generating executive summary: {str(e)}"

    def generate_key_findings(self, topic: str, sources: List[Source]) -> str:
        """
        Extract key findings from sources using LLM.

        Args:
            topic: Research topic
            sources: List of analyzed Source objects

        Returns:
            Key findings text (formatted list)

        Example:
            >>> agent = ReporterAgent(verbose=True)
            >>> findings = agent.generate_key_findings("climate change", sources)
        """
        self.log(f"Extracting key findings from {len(sources)} sources")

        try:
            # Create sources summary for prompt
            sources_summary = self._format_sources_for_prompt(sources)

            # Format prompt
            prompt = format_key_findings_prompt(topic=topic, sources_summary=sources_summary)

            # Get LLM response
            response = self.llm.invoke(prompt)

            # Extract content
            if hasattr(response, "content"):
                findings = response.content
            else:
                findings = str(response)

            self.log("Key findings extracted")
            return findings.strip()

        except Exception as e:
            self.log(f"Error extracting findings: {e}", level="ERROR")
            return f"Error extracting key findings: {str(e)}"

    def _format_sources_for_prompt(self, sources: List[Source]) -> str:
        """
        Format sources for LLM prompt.

        Args:
            sources: List of Source objects

        Returns:
            Formatted string summarizing sources
        """
        formatted = []
        for i, source in enumerate(sources, 1):
            preview = source.get_content_preview(300)
            formatted.append(
                f"{i}. [{source.title}]({source.url})\n"
                f"   Score: {source.trustworthiness_score:.1f}/100\n"
                f"   {preview}\n"
            )
        return "\n".join(formatted)

    def generate_html_report(
        self,
        topic: str,
        sources: List[Source],
        executive_summary: str,
        key_findings: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate complete HTML report.

        Args:
            topic: Research topic
            sources: List of analyzed Source objects
            executive_summary: Executive summary text
            key_findings: Key findings text
            metadata: Additional metadata for report

        Returns:
            HTML string

        Example:
            >>> agent = ReporterAgent()
            >>> html = agent.generate_html_report(
            ...     topic="Quantum Computing",
            ...     sources=sources,
            ...     executive_summary=summary,
            ...     key_findings=findings
            ... )
        """
        self.log("Generating HTML report")

        # Prepare metadata
        metadata = metadata or {}
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_sources = len(sources)
        trustworthy_count = len([s for s in sources if s.is_trustworthy()])
        avg_score = sum(s.trustworthiness_score for s in sources) / len(sources) if sources else 0

        # Limit sources if specified
        display_sources = sources
        if self.max_sources_in_report and len(sources) > self.max_sources_in_report:
            display_sources = sources[: self.max_sources_in_report]
            self.log(f"Limited report to top {self.max_sources_in_report} sources")

        # Generate HTML sections
        sources_html = self._generate_sources_section(display_sources)
        statistics_html = self._generate_statistics_section(sources)

        # Complete HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {topic}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Research Report</h1>
            <h2>{topic}</h2>
            <div class="metadata">
                <p>Generated: {generated_at}</p>
                <p>Total Sources: {total_sources} | Trustworthy: {trustworthy_count} | Avg Score: {avg_score:.1f}/100</p>
            </div>
        </header>

        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="content">
                {self._format_text_to_html(executive_summary)}
            </div>
        </section>

        <section class="key-findings">
            <h2>Key Findings</h2>
            <div class="content">
                {self._format_text_to_html(key_findings)}
            </div>
        </section>

        {statistics_html}

        <section class="sources">
            <h2>Sources</h2>
            {sources_html}
        </section>

        <footer>
            <p>Report generated by Research Assistant</p>
            <p class="timestamp">{generated_at}</p>
        </footer>
    </div>
</body>
</html>"""

        self.log("HTML report generated successfully")
        return html

    def _generate_sources_section(self, sources: List[Source]) -> str:
        """Generate HTML for sources listing."""
        sources_html = []

        for i, source in enumerate(sources, 1):
            # Determine trustworthiness class
            if source.trustworthiness_score >= 85:
                trust_class = "trust-high"
            elif source.trustworthiness_score >= 70:
                trust_class = "trust-medium"
            else:
                trust_class = "trust-low"

            # Get analysis metadata if available
            analysis = source.metadata.get("trustworthiness_analysis", {})
            reasoning = analysis.get("reasoning", "No analysis available")
            strengths = analysis.get("strengths", [])
            red_flags = analysis.get("red_flags", [])

            # Format content
            content_html = ""
            if self.include_full_content:
                content_html = f"""
                <div class="source-content">
                    <h4>Full Content</h4>
                    <p>{self._escape_html(source.content)}</p>
                </div>"""
            else:
                preview = source.get_content_preview(300)
                content_html = f"""
                <div class="source-preview">
                    <p>{self._escape_html(preview)}</p>
                </div>"""

            # Format strengths and red flags
            strengths_html = ""
            if strengths:
                strengths_list = "".join(f"<li>{self._escape_html(s)}</li>" for s in strengths)
                strengths_html = f"""
                <div class="strengths">
                    <strong>Strengths:</strong>
                    <ul>{strengths_list}</ul>
                </div>"""

            red_flags_html = ""
            if red_flags:
                flags_list = "".join(f"<li>{self._escape_html(f)}</li>" for f in red_flags)
                red_flags_html = f"""
                <div class="red-flags">
                    <strong>Concerns:</strong>
                    <ul>{flags_list}</ul>
                </div>"""

            # Metadata
            domain = source.get_domain()
            word_count = source.metadata.get("word_count", "N/A")

            source_html = f"""
            <div class="source {trust_class}">
                <div class="source-header">
                    <h3>{i}. {self._escape_html(source.title)}</h3>
                    <div class="trustworthiness-badge">
                        <span class="score">{source.trustworthiness_score:.1f}</span>/100
                    </div>
                </div>
                <div class="source-url">
                    <a href="{source.url}" target="_blank">{source.url}</a>
                </div>
                <div class="source-meta">
                    <span>Domain: {domain}</span> |
                    <span>Words: {word_count}</span>
                </div>
                {content_html}
                <div class="analysis">
                    <h4>Trustworthiness Analysis</h4>
                    <p>{self._escape_html(reasoning)}</p>
                    {strengths_html}
                    {red_flags_html}
                </div>
            </div>"""

            sources_html.append(source_html)

        return "\n".join(sources_html)

    def _generate_statistics_section(self, sources: List[Source]) -> str:
        """Generate HTML for statistics section."""
        if not sources:
            return ""

        scores = [s.trustworthiness_score for s in sources]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        # Score distribution
        excellent = len([s for s in scores if s >= 90])
        good = len([s for s in scores if 80 <= s < 90])
        fair = len([s for s in scores if 70 <= s < 80])
        poor = len([s for s in scores if s < 70])

        return f"""
        <section class="statistics">
            <h2>Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(sources)}</div>
                    <div class="stat-label">Total Sources</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_score:.1f}</div>
                    <div class="stat-label">Average Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{min_score:.1f} - {max_score:.1f}</div>
                    <div class="stat-label">Score Range</div>
                </div>
            </div>
            <div class="distribution">
                <h3>Score Distribution</h3>
                <ul>
                    <li><span class="trust-high">■</span> Excellent (90-100): {excellent}</li>
                    <li><span class="trust-medium">■</span> Good (80-89): {good}</li>
                    <li><span class="trust-medium">■</span> Fair (70-79): {fair}</li>
                    <li><span class="trust-low">■</span> Poor (0-69): {poor}</li>
                </ul>
            </div>
        </section>"""

    def _format_text_to_html(self, text: str) -> str:
        """Convert plain text to HTML with paragraph breaks."""
        # Split by double newlines for paragraphs
        paragraphs = text.split("\n\n")
        html_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if para:
                # Check if it's a list item
                if para.startswith(
                    ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "-", "*")
                ):
                    # Keep as-is for lists (will be wrapped in content div)
                    html_paragraphs.append(f"<p>{self._escape_html(para)}</p>")
                else:
                    html_paragraphs.append(f"<p>{self._escape_html(para)}</p>")

        return "\n".join(html_paragraphs)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _get_css_styles(self) -> str:
        """Return CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        header h2 {
            font-size: 1.8em;
            font-weight: normal;
            opacity: 0.9;
        }

        .metadata {
            margin-top: 20px;
            opacity: 0.8;
            font-size: 0.9em;
        }

        section {
            margin-bottom: 40px;
            padding: 30px;
            background-color: #fafafa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }

        .content {
            line-height: 1.8;
        }

        .content p {
            margin-bottom: 15px;
        }

        .source {
            background-color: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .source-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }

        .source h3 {
            color: #333;
            font-size: 1.3em;
            flex: 1;
        }

        .trustworthiness-badge {
            background-color: #f0f0f0;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 15px;
        }

        .trust-high .trustworthiness-badge {
            background-color: #d4edda;
            color: #155724;
        }

        .trust-medium .trustworthiness-badge {
            background-color: #fff3cd;
            color: #856404;
        }

        .trust-low .trustworthiness-badge {
            background-color: #f8d7da;
            color: #721c24;
        }

        .source-url {
            margin-bottom: 10px;
        }

        .source-url a {
            color: #667eea;
            text-decoration: none;
            word-break: break-all;
        }

        .source-url a:hover {
            text-decoration: underline;
        }

        .source-meta {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 15px;
        }

        .source-preview, .source-content {
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 3px solid #667eea;
            margin: 15px 0;
        }

        .analysis {
            margin-top: 20px;
            padding: 15px;
            background-color: #f0f4ff;
            border-radius: 5px;
        }

        .analysis h4 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .strengths, .red-flags {
            margin-top: 10px;
        }

        .strengths ul, .red-flags ul {
            margin-left: 20px;
            margin-top: 5px;
        }

        .strengths strong {
            color: #28a745;
        }

        .red-flags strong {
            color: #dc3545;
        }

        .statistics {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .stat-card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }

        .stat-label {
            color: #666;
            margin-top: 5px;
        }

        .distribution {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
        }

        .distribution h3 {
            color: #667eea;
            margin-bottom: 15px;
        }

        .distribution ul {
            list-style: none;
        }

        .distribution li {
            padding: 8px 0;
            font-size: 1.1em;
        }

        footer {
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 1px solid #ddd;
        }

        .timestamp {
            margin-top: 10px;
            font-size: 0.9em;
        }

        @media print {
            body {
                background-color: white;
            }
            .source {
                page-break-inside: avoid;
            }
        }
        """

    def run(
        self, topic: str, sources: List[Source], output_path: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute report generation workflow.

        Args:
            topic: Research topic
            sources: List of analyzed Source objects
            output_path: Optional path to save HTML file
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - success: bool
                - html_report: str (HTML content)
                - output_path: Optional[str]
                - sources_count: int
                - error: Optional[str]

        Example:
            >>> agent = ReporterAgent(verbose=True)
            >>> result = agent.run(
            ...     topic="Machine Learning",
            ...     sources=analyzed_sources,
            ...     output_path="report.html"
            ... )
            >>> print(result['html_report'][:100])
        """
        try:
            # Validate inputs
            if not sources:
                return self.handle_error(
                    ValueError("No sources provided"), context="ReporterAgent.run"
                )

            if not topic:
                return self.handle_error(
                    ValueError("Research topic required"), context="ReporterAgent.run"
                )

            # Filter trustworthy sources for analysis
            trustworthy_sources = [s for s in sources if s.trustworthiness_score >= 85.0]

            # Use trustworthy sources for executive summary and findings
            # Use all sources for full report (for transparency)
            self.log(
                f"Using {len(trustworthy_sources)}/{len(sources)} trustworthy sources for analysis"
            )

            # Generate report components
            if trustworthy_sources:
                self.log("Generating executive summary...")
                executive_summary = self.generate_executive_summary(topic, trustworthy_sources)

                self.log("Extracting key findings...")
                key_findings = self.generate_key_findings(topic, trustworthy_sources)
            else:
                # No trustworthy sources - generate warning message
                self.log(
                    "No trustworthy sources found - generating limited report", level="WARNING"
                )
                executive_summary = (
                    f"No trustworthy sources (score >= 85) were found for this research topic. "
                    f"All {len(sources)} sources analyzed had trustworthiness scores below the threshold. "
                    f"Please review the sources below with caution and consider conducting additional research "
                    f"from more authoritative sources."
                )
                key_findings = (
                    "- No reliable key findings could be extracted due to low source quality.\n"
                    "- Further research from credible sources is recommended.\n"
                    f"- Review the {len(sources)} sources below for potential leads."
                )

            self.log("Generating HTML report...")
            html_report = self.generate_html_report(
                topic=topic,
                sources=sources,  # Include ALL sources in full report
                executive_summary=executive_summary,
                key_findings=key_findings,
            )

            # Optionally save to file
            if output_path:
                self.log(f"Saving report to {output_path}")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_report)
                self.log("Report saved successfully")

            # Return results
            return self.create_success_result(
                {
                    "html_report": html_report,
                    "output_path": output_path,
                    "sources_count": len(sources),
                    "executive_summary": executive_summary,
                    "key_findings": key_findings,
                }
            )

        except Exception as e:
            return self.handle_error(e, context="ReporterAgent.run")
