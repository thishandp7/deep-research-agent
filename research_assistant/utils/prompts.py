"""
Prompt templates for LLM interactions.

Contains all prompt templates used by agents in the research assistant.
"""

from langchain.prompts import PromptTemplate


# ============================================================================
# Query Generation Prompts
# ============================================================================

QUERY_GENERATION_TEMPLATE = """You are a research assistant helping to explore a topic thoroughly.

Given a research topic, generate 3-5 diverse search queries that will help discover comprehensive information from different angles.

Topic: {topic}

Guidelines:
- Create queries that explore different aspects (what, why, how, history, current state, future)
- Use varied phrasing to discover different sources
- Keep queries concise but specific
- Avoid overly broad or overly narrow queries

Generate search queries (one per line):"""

query_generation_prompt = PromptTemplate(
    input_variables=["topic"],
    template=QUERY_GENERATION_TEMPLATE
)


# ============================================================================
# Trustworthiness Analysis Prompts
# ============================================================================

TRUSTWORTHINESS_ANALYSIS_TEMPLATE = """You are an expert fact-checker and source evaluator.

Analyze the following web source for trustworthiness on a scale of 0-100.

Research Topic: {topic}
Source URL: {url}
Source Title: {title}
Content Preview: {content_preview}

Evaluation Criteria:
1. **Content Quality (30%)**: Coherence, depth, citations, evidence
2. **Bias Detection (20%)**: Objectivity vs propaganda, balanced presentation
3. **Factual Density (15%)**: Ratio of verifiable facts to opinions
4. **Source Credibility (20%)**: Domain authority, author credentials
5. **Relevance (15%)**: How well it addresses the research topic

Analyze each criterion and provide:
1. A trustworthiness score (0-100, where 100 is most trustworthy)
2. Brief reasoning for your score
3. Any red flags or concerns

Return ONLY a JSON object in this exact format:
{{
  "score": <number 0-100>,
  "reasoning": "<2-3 sentence explanation>",
  "red_flags": ["<concern 1>", "<concern 2>"],
  "strengths": ["<strength 1>", "<strength 2>"]
}}"""

trustworthiness_analysis_prompt = PromptTemplate(
    input_variables=["topic", "url", "title", "content_preview"],
    template=TRUSTWORTHINESS_ANALYSIS_TEMPLATE
)


# ============================================================================
# Report Generation Prompts
# ============================================================================

REPORT_SUMMARY_TEMPLATE = """You are a research analyst creating an executive summary.

Research Topic: {topic}

Based on the following {num_sources} trustworthy sources, create a comprehensive but concise executive summary (3-5 paragraphs).

Sources:
{sources_summary}

Guidelines:
- Start with a clear definition or overview
- Highlight key findings and themes across sources
- Note any consensus or disagreements among sources
- Keep it factual and objective
- Write in clear, accessible language

Executive Summary:"""

report_summary_prompt = PromptTemplate(
    input_variables=["topic", "num_sources", "sources_summary"],
    template=REPORT_SUMMARY_TEMPLATE
)


REPORT_KEY_FINDINGS_TEMPLATE = """Extract 5-7 key findings from the research on: {topic}

Based on these sources:
{sources_summary}

For each finding:
- State it clearly and concisely
- Note which source(s) support it
- Indicate confidence level if applicable

Format as a numbered list.

Key Findings:"""

report_key_findings_prompt = PromptTemplate(
    input_variables=["topic", "sources_summary"],
    template=REPORT_KEY_FINDINGS_TEMPLATE
)


# ============================================================================
# Content Extraction Prompts
# ============================================================================

CONTENT_SUMMARY_TEMPLATE = """Summarize the main points from this web content in 2-3 sentences.

Content: {content}

Summary:"""

content_summary_prompt = PromptTemplate(
    input_variables=["content"],
    template=CONTENT_SUMMARY_TEMPLATE
)


# ============================================================================
# Helper Functions
# ============================================================================

def format_query_generation_prompt(topic: str) -> str:
    """
    Format query generation prompt.

    Args:
        topic: Research topic

    Returns:
        Formatted prompt string
    """
    return query_generation_prompt.format(topic=topic)


def format_trustworthiness_prompt(
    topic: str,
    url: str,
    title: str,
    content_preview: str
) -> str:
    """
    Format trustworthiness analysis prompt.

    Args:
        topic: Research topic
        url: Source URL
        title: Source title
        content_preview: Preview of source content

    Returns:
        Formatted prompt string
    """
    return trustworthiness_analysis_prompt.format(
        topic=topic,
        url=url,
        title=title,
        content_preview=content_preview
    )


def format_report_summary_prompt(
    topic: str,
    sources_summary: str,
    num_sources: int
) -> str:
    """
    Format report summary generation prompt.

    Args:
        topic: Research topic
        sources_summary: Summary of all sources
        num_sources: Number of sources

    Returns:
        Formatted prompt string
    """
    return report_summary_prompt.format(
        topic=topic,
        sources_summary=sources_summary,
        num_sources=num_sources
    )


def format_key_findings_prompt(topic: str, sources_summary: str) -> str:
    """
    Format key findings extraction prompt.

    Args:
        topic: Research topic
        sources_summary: Summary of all sources

    Returns:
        Formatted prompt string
    """
    return report_key_findings_prompt.format(
        topic=topic,
        sources_summary=sources_summary
    )


# ============================================================================
# Prompt Variations (for future experimentation)
# ============================================================================

# Alternative trustworthiness prompt (more structured)
TRUSTWORTHINESS_STRUCTURED_TEMPLATE = """Evaluate source trustworthiness (0-100):

Topic: {topic}
URL: {url}
Title: {title}

Rate each factor (0-100):
1. Content Quality: [score]
2. Objectivity: [score]
3. Evidence Strength: [score]
4. Source Authority: [score]

Overall Score: [weighted average]
Reasoning: [explanation]
"""

# Alternative query generation (broader)
QUERY_GENERATION_BROAD_TEMPLATE = """Generate search queries for: {topic}

Create 5 queries covering:
1. Basic definition/overview
2. Historical context
3. Current applications
4. Challenges/criticisms
5. Future developments

Queries:"""


# ============================================================================
# Prompt Constants
# ============================================================================

# Default number of queries to generate
DEFAULT_NUM_QUERIES = 5

# Default content preview length for analysis
DEFAULT_PREVIEW_LENGTH = 500

# Temperature settings for different tasks
TEMPERATURE_CREATIVE = 0.9  # For brainstorming, query generation
TEMPERATURE_BALANCED = 0.7  # For general tasks
TEMPERATURE_ANALYTICAL = 0.3  # For factual analysis, scoring
