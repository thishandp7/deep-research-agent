"""
HTML Report Generation with Jinja2

Professional HTML report templates for research results.
"""

from datetime import datetime
from typing import List, Optional
from jinja2 import Template

from research_assistant.models.source import Source


# ============================================================================
# Jinja2 Template
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {{ topic }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        header h2 {
            font-size: 1.8em;
            font-weight: 400;
            opacity: 0.95;
        }

        .metadata {
            margin-top: 20px;
            font-size: 0.95em;
            opacity: 0.9;
        }

        .metadata p {
            margin: 5px 0;
        }

        .content-wrapper {
            padding: 40px;
        }

        section {
            margin-bottom: 40px;
        }

        section h2 {
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .executive-summary .content,
        .key-findings .content {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .executive-summary p,
        .key-findings p {
            margin-bottom: 15px;
        }

        .key-findings ul {
            list-style: none;
            padding-left: 0;
        }

        .key-findings li {
            padding: 10px 0;
            padding-left: 30px;
            position: relative;
        }

        .key-findings li:before {
            content: "▸";
            position: absolute;
            left: 10px;
            color: #667eea;
            font-weight: bold;
        }

        .statistics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-card .label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .source-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .source-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .source-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }

        .source-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
            flex: 1;
        }

        .source-title a {
            color: #667eea;
            text-decoration: none;
        }

        .source-title a:hover {
            text-decoration: underline;
        }

        .trust-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
            white-space: nowrap;
            margin-left: 15px;
        }

        .trust-high {
            background: #d4edda;
            color: #155724;
        }

        .trust-medium {
            background: #fff3cd;
            color: #856404;
        }

        .trust-low {
            background: #f8d7da;
            color: #721c24;
        }

        .source-url {
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 10px;
            word-break: break-all;
        }

        .source-content {
            color: #555;
            line-height: 1.8;
            margin: 15px 0;
        }

        .source-metadata {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
            font-size: 0.9em;
            color: #6c757d;
        }

        .metadata-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .metadata-label {
            font-weight: 600;
        }

        .analysis-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
        }

        .analysis-section h4 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .strengths, .red-flags {
            list-style: none;
            padding-left: 0;
        }

        .strengths li, .red-flags li {
            padding: 5px 0;
            padding-left: 20px;
            position: relative;
        }

        .strengths li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #28a745;
            font-weight: bold;
        }

        .red-flags li:before {
            content: "⚠";
            position: absolute;
            left: 0;
            color: #dc3545;
        }

        footer {
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #e0e0e0;
        }

        footer p {
            margin: 5px 0;
        }

        .timestamp {
            font-size: 0.9em;
            opacity: 0.8;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }

            .container {
                box-shadow: none;
            }

            .source-card {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Research Report</h1>
            <h2>{{ topic }}</h2>
            <div class="metadata">
                <p>Generated: {{ generated_at }}</p>
                <p>Total Sources: {{ total_sources }} | Trustworthy: {{ trustworthy_count }} | Average Score: {{ avg_score }}/100</p>
            </div>
        </header>

        <div class="content-wrapper">
            <!-- Executive Summary -->
            <section class="executive-summary">
                <h2>Executive Summary</h2>
                <div class="content">
                    {{ executive_summary | safe }}
                </div>
            </section>

            <!-- Key Findings -->
            <section class="key-findings">
                <h2>Key Findings</h2>
                <div class="content">
                    {{ key_findings | safe }}
                </div>
            </section>

            <!-- Statistics -->
            {% if show_statistics %}
            <section class="statistics-section">
                <h2>Research Statistics</h2>
                <div class="statistics">
                    <div class="stat-card">
                        <div class="value">{{ total_sources }}</div>
                        <div class="label">Total Sources</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{{ trustworthy_count }}</div>
                        <div class="label">Trustworthy Sources</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{{ avg_score }}</div>
                        <div class="label">Average Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{{ queries_count }}</div>
                        <div class="label">Search Queries</div>
                    </div>
                </div>
            </section>
            {% endif %}

            <!-- Sources -->
            <section class="sources">
                <h2>Sources ({{ sources|length }})</h2>
                {% for source in sources %}
                <div class="source-card">
                    <div class="source-header">
                        <div class="source-title">
                            <a href="{{ source.url }}" target="_blank" rel="noopener">{{ source.title }}</a>
                        </div>
                        <span class="trust-badge {{ source.trust_class }}">
                            {{ source.trustworthiness_score }}/100
                        </span>
                    </div>

                    <div class="source-url">{{ source.url }}</div>

                    {% if source.content_preview %}
                    <div class="source-content">
                        {{ source.content_preview }}
                    </div>
                    {% endif %}

                    <div class="source-metadata">
                        {% if source.domain %}
                        <div class="metadata-item">
                            <span class="metadata-label">Domain:</span>
                            <span>{{ source.domain }}</span>
                        </div>
                        {% endif %}
                        {% if source.word_count %}
                        <div class="metadata-item">
                            <span class="metadata-label">Words:</span>
                            <span>{{ source.word_count }}</span>
                        </div>
                        {% endif %}
                        {% if source.published_date %}
                        <div class="metadata-item">
                            <span class="metadata-label">Published:</span>
                            <span>{{ source.published_date }}</span>
                        </div>
                        {% endif %}
                    </div>

                    {% if source.analysis %}
                    <div class="analysis-section">
                        <h4>Trustworthiness Analysis</h4>
                        <p><strong>Reasoning:</strong> {{ source.analysis.reasoning }}</p>

                        {% if source.analysis.strengths %}
                        <p><strong>Strengths:</strong></p>
                        <ul class="strengths">
                            {% for strength in source.analysis.strengths %}
                            <li>{{ strength }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}

                        {% if source.analysis.red_flags %}
                        <p><strong>Red Flags:</strong></p>
                        <ul class="red-flags">
                            {% for flag in source.analysis.red_flags %}
                            <li>{{ flag }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </section>
        </div>

        <footer>
            <p><strong>Deep Research Assistant</strong></p>
            <p>AI-powered research with trustworthiness analysis</p>
            <p class="timestamp">{{ generated_at }}</p>
        </footer>
    </div>
</body>
</html>
"""


# ============================================================================
# HTML Generator Class
# ============================================================================


class HTMLReportGenerator:
    """
    Generate professional HTML reports using Jinja2 templates.

    Example:
        >>> generator = HTMLReportGenerator()
        >>> html = generator.generate(
        ...     topic="Machine Learning",
        ...     sources=analyzed_sources,
        ...     executive_summary="Summary text",
        ...     key_findings="Findings text"
        ... )
    """

    def __init__(self, template: Optional[str] = None):
        """
        Initialize HTML generator.

        Args:
            template: Optional custom Jinja2 template string
        """
        self.template = Template(template or HTML_TEMPLATE)

    def generate(
        self,
        topic: str,
        sources: List[Source],
        executive_summary: str,
        key_findings: str,
        queries_count: int = 0,
        show_statistics: bool = True,
    ) -> str:
        """
        Generate HTML report from research data.

        Args:
            topic: Research topic
            sources: List of analyzed Source objects
            executive_summary: Executive summary text
            key_findings: Key findings text
            queries_count: Number of search queries generated
            show_statistics: Whether to show statistics section

        Returns:
            Complete HTML document as string

        Example:
            >>> generator = HTMLReportGenerator()
            >>> html = generator.generate(
            ...     topic="AI Safety",
            ...     sources=sources,
            ...     executive_summary="...",
            ...     key_findings="..."
            ... )
        """
        # Calculate statistics
        total_sources = len(sources)
        trustworthy_count = len([s for s in sources if s.trustworthiness_score >= 85.0])
        avg_score = (
            f"{sum(s.trustworthiness_score for s in sources) / len(sources):.1f}"
            if sources
            else "0.0"
        )
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare source data for template
        template_sources = []
        for source in sources:
            # Determine trust class
            if source.trustworthiness_score >= 85:
                trust_class = "trust-high"
            elif source.trustworthiness_score >= 70:
                trust_class = "trust-medium"
            else:
                trust_class = "trust-low"

            # Get analysis data
            analysis_data = source.metadata.get("trustworthiness_analysis", {})

            # Format source for template
            template_sources.append(
                {
                    "url": source.url,
                    "title": source.title,
                    "trustworthiness_score": f"{source.trustworthiness_score:.1f}",
                    "trust_class": trust_class,
                    "content_preview": source.get_content_preview(400),
                    "domain": source.metadata.get("domain", ""),
                    "word_count": source.metadata.get("word_count", ""),
                    "published_date": source.metadata.get("published_date", ""),
                    "analysis": (
                        {
                            "reasoning": analysis_data.get("reasoning", ""),
                            "strengths": analysis_data.get("strengths", []),
                            "red_flags": analysis_data.get("red_flags", []),
                        }
                        if analysis_data
                        else None
                    ),
                }
            )

        # Format text content for HTML
        executive_summary_html = self._format_text_to_html(executive_summary)
        key_findings_html = self._format_text_to_html(key_findings)

        # Render template
        return self.template.render(
            topic=topic,
            generated_at=generated_at,
            total_sources=total_sources,
            trustworthy_count=trustworthy_count,
            avg_score=avg_score,
            queries_count=queries_count,
            executive_summary=executive_summary_html,
            key_findings=key_findings_html,
            sources=template_sources,
            show_statistics=show_statistics,
        )

    def _format_text_to_html(self, text: str) -> str:
        """
        Convert plain text to HTML with basic formatting.

        Args:
            text: Plain text

        Returns:
            HTML formatted text
        """
        if not text:
            return "<p>No content available.</p>"

        # Split into paragraphs
        paragraphs = text.strip().split("\n\n")

        html_parts = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if it's a list (starts with -, *, or number.)
            lines = para.split("\n")
            if len(lines) > 1 and (lines[0].strip().startswith(("-", "*", "1.", "2.", "3."))):
                # Convert to HTML list
                items = []
                for line in lines:
                    line = line.strip()
                    if line.startswith(("-", "*")):
                        items.append(f"<li>{line[1:].strip()}</li>")
                    elif line and line[0].isdigit():
                        # Remove number prefix
                        items.append(f"<li>{line.split('.', 1)[1].strip()}</li>")
                    elif line:
                        items.append(f"<li>{line}</li>")

                html_parts.append(f"<ul>{''.join(items)}</ul>")
            else:
                # Regular paragraph
                html_parts.append(f"<p>{para}</p>")

        return "\n".join(html_parts)


# ============================================================================
# Convenience Functions
# ============================================================================


def generate_html_report(
    topic: str,
    sources: List[Source],
    executive_summary: str,
    key_findings: str,
    queries_count: int = 0,
) -> str:
    """
    Convenience function to generate HTML report.

    Args:
        topic: Research topic
        sources: List of analyzed sources
        executive_summary: Executive summary text
        key_findings: Key findings text
        queries_count: Number of queries generated

    Returns:
        HTML report string

    Example:
        >>> html = generate_html_report(
        ...     topic="Climate Change",
        ...     sources=sources,
        ...     executive_summary="...",
        ...     key_findings="..."
        ... )
    """
    generator = HTMLReportGenerator()
    return generator.generate(
        topic=topic,
        sources=sources,
        executive_summary=executive_summary,
        key_findings=key_findings,
        queries_count=queries_count,
    )
