"""
Tests for HTML report generator.
"""

from research_assistant.utils.html_generator import (
    HTMLReportGenerator,
    generate_html_report,
)
from research_assistant.models.source import Source


class TestHTMLReportGenerator:
    """Test HTMLReportGenerator class."""

    def test_generates_basic_html_report(self):
        """Test that basic HTML report is generated."""
        generator = HTMLReportGenerator()

        source = Source(
            url="https://example.com/article",
            title="Test Article",
            content="This is test content. " * 50,
            trustworthiness_score=90.0,
            metadata={"domain": "example.com"},
        )

        html = generator.generate(
            topic="Test Topic",
            sources=[source],
            executive_summary="This is a test summary.",
            key_findings="- Finding 1\n- Finding 2",
        )

        assert "<!DOCTYPE html>" in html
        assert "Test Topic" in html
        assert "Test Article" in html
        assert "example.com/article" in html

    def test_report_contains_all_sections(self):
        """Test that report contains all required sections."""
        generator = HTMLReportGenerator()

        source = Source(
            url="https://example.com/article",
            title="Article",
            content="Content here.",
            trustworthiness_score=85.0,
        )

        html = generator.generate(
            topic="Topic",
            sources=[source],
            executive_summary="Summary",
            key_findings="Findings",
        )

        assert "Executive Summary" in html
        assert "Key Findings" in html
        assert "Sources" in html

    def test_classifies_source_trustworthiness(self):
        """Test that sources are classified by trustworthiness."""
        generator = HTMLReportGenerator()

        high_trust = Source(
            url="https://high.com", title="High", content="x" * 100, trustworthiness_score=90.0
        )
        medium_trust = Source(
            url="https://medium.com",
            title="Medium",
            content="x" * 100,
            trustworthiness_score=75.0,
        )
        low_trust = Source(
            url="https://low.com", title="Low", content="x" * 100, trustworthiness_score=50.0
        )

        html = generator.generate(
            topic="Topic",
            sources=[high_trust, medium_trust, low_trust],
            executive_summary="Summary",
            key_findings="Findings",
        )

        assert "trust-high" in html
        assert "trust-medium" in html
        assert "trust-low" in html

    def test_includes_source_metadata(self):
        """Test that source metadata is included."""
        generator = HTMLReportGenerator()

        source = Source(
            url="https://example.com",
            title="Article",
            content="Content",
            trustworthiness_score=85.0,
            metadata={
                "domain": "example.com",
                "word_count": 500,
                "published_date": "2024-01-01",
            },
        )

        html = generator.generate(
            topic="Topic",
            sources=[source],
            executive_summary="Summary",
            key_findings="Findings",
        )

        assert "example.com" in html
        assert "500" in html
        assert "2024-01-01" in html

    def test_includes_trustworthiness_analysis(self):
        """Test that trustworthiness analysis is displayed."""
        generator = HTMLReportGenerator()

        source = Source(
            url="https://example.com",
            title="Article",
            content="Content",
            trustworthiness_score=88.0,
            metadata={
                "trustworthiness_analysis": {
                    "reasoning": "Well-researched article",
                    "strengths": ["Good citations", "Clear evidence"],
                    "red_flags": [],
                }
            },
        )

        html = generator.generate(
            topic="Topic",
            sources=[source],
            executive_summary="Summary",
            key_findings="Findings",
        )

        assert "Trustworthiness Analysis" in html
        assert "Well-researched article" in html
        assert "Good citations" in html
        assert "Clear evidence" in html

    def test_handles_empty_sources_list(self):
        """Test handling of empty sources list."""
        generator = HTMLReportGenerator()

        html = generator.generate(
            topic="Topic",
            sources=[],
            executive_summary="No sources found",
            key_findings="N/A",
        )

        assert "<!DOCTYPE html>" in html
        assert "Topic" in html

    def test_calculates_statistics_correctly(self):
        """Test that statistics are calculated correctly."""
        generator = HTMLReportGenerator()

        sources = [
            Source(
                url=f"https://{i}.com",
                title=f"S{i}",
                content="x" * 100,
                trustworthiness_score=score,
            )
            for i, score in enumerate([90, 85, 80, 75, 70])
        ]

        html = generator.generate(
            topic="Topic",
            sources=sources,
            executive_summary="Summary",
            key_findings="Findings",
            queries_count=5,
        )

        # Total sources: 5
        assert "Total Sources: 5" in html or ">5<" in html

        # Trustworthy (>= 85): 2 sources
        assert "Trustworthy: 2" in html or ">2<" in html

        # Average: (90+85+80+75+70)/5 = 80.0
        assert "80.0" in html or "80" in html

    def test_formats_paragraphs_correctly(self):
        """Test that text formatting works."""
        generator = HTMLReportGenerator()

        source = Source(
            url="https://example.com",
            title="Article",
            content="Content",
            trustworthiness_score=85.0,
        )

        html = generator.generate(
            topic="Topic",
            sources=[source],
            executive_summary="Paragraph 1.\n\nParagraph 2.",
            key_findings="Findings",
        )

        assert "<p>Paragraph 1.</p>" in html
        assert "<p>Paragraph 2.</p>" in html

    def test_formats_lists_correctly(self):
        """Test that lists are formatted as HTML lists."""
        generator = HTMLReportGenerator()

        source = Source(
            url="https://example.com",
            title="Article",
            content="Content",
            trustworthiness_score=85.0,
        )

        html = generator.generate(
            topic="Topic",
            sources=[source],
            executive_summary="Summary",
            key_findings="- Finding 1\n- Finding 2\n- Finding 3",
        )

        assert "<ul>" in html or "<li>" in html


class TestConvenienceFunction:
    """Test convenience function."""

    def test_generate_html_report_function(self):
        """Test generate_html_report convenience function."""
        source = Source(
            url="https://example.com",
            title="Article",
            content="Content",
            trustworthiness_score=85.0,
        )

        html = generate_html_report(
            topic="Topic",
            sources=[source],
            executive_summary="Summary",
            key_findings="Findings",
        )

        assert "<!DOCTYPE html>" in html
        assert "Topic" in html
        assert "Article" in html


class TestCustomTemplate:
    """Test custom template support."""

    def test_accepts_custom_template(self):
        """Test that generator accepts custom template."""
        custom_template = """
        <html>
        <body>
        <h1>{{ topic }}</h1>
        <p>Custom template</p>
        </body>
        </html>
        """

        generator = HTMLReportGenerator(template=custom_template)

        source = Source(
            url="https://example.com",
            title="Article",
            content="Content",
            trustworthiness_score=85.0,
        )

        html = generator.generate(
            topic="My Topic",
            sources=[source],
            executive_summary="Summary",
            key_findings="Findings",
        )

        assert "My Topic" in html
        assert "Custom template" in html
