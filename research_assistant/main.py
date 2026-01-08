#!/usr/bin/env python3
"""
Research Assistant CLI

Command-line interface for the Deep Research Assistant.
Orchestrates the LangGraph workflow and generates research reports.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich import box

from research_assistant.graph import create_research_graph
from research_assistant.graph.state import create_initial_state
from research_assistant.config import settings


console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Research Assistant - AI-powered research with trustworthiness analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic research with default settings
  %(prog)s "artificial general intelligence"

  # Specify number of sources and output location
  %(prog)s "quantum computing" --max-sources 15 --output reports/quantum.html

  # Use specific LLM model
  %(prog)s "climate change" --model llama3.1:8b

  # Verbose mode with detailed logging
  %(prog)s "machine learning" --verbose
        """,
    )

    parser.add_argument(
        "topic",
        type=str,
        help="Research topic to investigate",
    )

    parser.add_argument(
        "--max-sources",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of sources to process (default: 10)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for HTML report (default: auto-generated in data/reports/)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="LLM model to use (default: from config)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Research Assistant v0.1.0",
    )

    return parser.parse_args()


def generate_output_path(topic: str) -> Path:
    """Generate output path for HTML report."""
    # Clean topic for filename
    safe_topic = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in topic)
    safe_topic = safe_topic.replace(" ", "_").lower()

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.html"

    # Ensure reports directory exists
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    return reports_dir / filename


def display_header(topic: str, max_sources: int):
    """Display welcome header."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Deep Research Assistant[/bold cyan]\n"
            "[dim]AI-powered research with trustworthiness analysis[/dim]",
            border_style="cyan",
        )
    )
    console.print()
    console.print(f"[bold]Research Topic:[/bold] {topic}")
    console.print(f"[bold]Max Sources:[/bold] {max_sources}")
    console.print()


def display_results(final_state: dict, output_path: Path):
    """Display research results."""
    console.print()
    console.print("[bold green]✓ Research Complete![/bold green]")
    console.print()

    # Create results table
    table = Table(title="Research Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Add statistics
    table.add_row("Search Queries Generated", str(len(final_state.get("search_queries", []))))
    table.add_row("URLs Discovered", str(len(final_state.get("discovered_urls", []))))
    table.add_row("Sources Scraped", str(len(final_state.get("scraped_sources", []))))
    table.add_row("Failed URLs", str(len(final_state.get("failed_urls", []))))

    analyzed = final_state.get("analyzed_sources", [])
    if analyzed:
        scores = [s.trustworthiness_score for s in analyzed]
        avg_score = sum(scores) / len(scores)
        table.add_row("Sources Analyzed", str(len(analyzed)))
        table.add_row("Average Trust Score", f"{avg_score:.1f}/100")
        table.add_row("Score Range", f"{min(scores):.1f} - {max(scores):.1f}")

    table.add_row("Trustworthy Sources Stored", str(len(final_state.get("stored_sources", []))))
    table.add_row("Low-Quality Sources", str(len(final_state.get("rejected_sources", []))))

    console.print(table)
    console.print()

    # Display errors if any
    errors = final_state.get("errors", [])
    if errors:
        console.print("[yellow]⚠ Warnings/Errors encountered:[/yellow]")
        for error in errors:
            console.print(f"  [dim]• {error}[/dim]")
        console.print()

    # Display output path
    console.print(
        f"[bold green]Report saved to:[/bold green] [link=file://{output_path.absolute()}]{output_path}[/link]"
    )
    console.print()


def run_research_workflow(topic: str, max_sources: int, verbose: bool, show_progress: bool) -> dict:
    """
    Run the research workflow with progress tracking.

    Args:
        topic: Research topic
        max_sources: Maximum sources to process
        verbose: Enable verbose logging
        show_progress: Show progress bars

    Returns:
        Final research state
    """
    # Create initial state
    initial_state = create_initial_state(topic=topic, max_sources=max_sources)

    # Create graph
    graph = create_research_graph(verbose=verbose)

    if not show_progress:
        # Simple execution without progress bars
        console.print("[dim]Running research workflow...[/dim]")
        return graph.invoke(initial_state)

    # Execute with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Add tasks
        task = progress.add_task("[cyan]Running research workflow...", total=6)

        # Note: This is a simplified progress display
        # In reality, the graph runs atomically, so we update all at once
        # For true progress tracking, we'd need to instrument individual nodes

        final_state = graph.invoke(initial_state)

        # Update progress to complete
        progress.update(task, completed=6)

    return final_state


def main():
    """Main CLI entry point."""
    # Parse arguments
    args = parse_args()

    # Update config if model specified
    if args.model:
        settings.llm_model = args.model

    # Generate output path
    output_path = Path(args.output) if args.output else generate_output_path(args.topic)

    # Display header
    display_header(args.topic, args.max_sources)

    try:
        # Run research workflow
        final_state = run_research_workflow(
            topic=args.topic,
            max_sources=args.max_sources,
            verbose=args.verbose,
            show_progress=not args.no_progress,
        )

        # Save report
        if final_state.get("report_html"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_state["report_html"])

        # Display results
        display_results(final_state, output_path)

        # Exit successfully
        sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/yellow]")
        sys.exit(130)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
