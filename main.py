#!/usr/bin/env python3
"""
EV Subsidy Analytics Pipeline - Main Entry Point
Demonstrates end-to-end data engineering pipeline for MIT MBAn interview

Usage:
    python main.py              # Run full pipeline
    python main.py --data-only  # Generate data only
    python main.py --api        # Start API server
    python main.py --viz        # Generate visualizations only
"""

import argparse
import json
import sys
import time
from pathlib import Path


def run_data_pipeline():
    """Run the data generation and normalization pipeline."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA GENERATION")
    print("=" * 60)

    from src.pipeline.data_generator import EVSubsidyDataGenerator

    generator = EVSubsidyDataGenerator(seed=42)
    files = generator.save_raw_data()

    print(f"Generated raw data files:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    print("\n" + "=" * 60)
    print("STEP 2: DATA NORMALIZATION")
    print("=" * 60)

    from src.pipeline.normalizer import normalize_pipeline

    df, metrics = normalize_pipeline()

    print(f"\nNormalization complete:")
    print(f"  Records processed: {metrics['records_processed']}")
    print(f"  Records normalized: {metrics['records_normalized']}")
    print(f"  Processing time: {metrics['processing_time_ms']}ms")

    print("\n" + "=" * 60)
    print("STEP 3: AI SUMMARIZATION")
    print("=" * 60)

    from src.pipeline.summarizer import PolicySummarizer

    # Use rule-based for demo (faster, no model download required)
    summarizer = PolicySummarizer(use_ai=False)
    df_summarized = summarizer.summarize_all_policies(df)
    summary_files = summarizer.save_summaries(df_summarized)

    print(f"Generated summary files:")
    for name, path in summary_files.items():
        print(f"  {name}: {path}")

    return df_summarized, metrics


def run_visualizations():
    """Generate all visualizations and slides."""
    print("\n" + "=" * 60)
    print("STEP 4: VISUALIZATION GENERATION")
    print("=" * 60)

    from src.visualization.charts import EVSubsidyVisualizer
    from src.visualization.slides import SlideGenerator

    # Generate charts
    visualizer = EVSubsidyVisualizer()
    visualizer.load_data()
    charts = visualizer.create_all_charts()

    print(f"\nGenerated {len(charts)} charts:")
    for name, path in charts.items():
        print(f"  {name}: {path}")

    # Generate slides
    print("\nGenerating presentation slides...")
    slide_gen = SlideGenerator()
    slides = slide_gen.create_all_slides()

    print(f"\nGenerated {len(slides)} slides:")
    for name, path in slides.items():
        print(f"  {name}: {path}")

    return charts, slides


def run_api_server():
    """Start the FastAPI server."""
    print("\n" + "=" * 60)
    print("STARTING API SERVER")
    print("=" * 60)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")

    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


def print_summary(metrics: dict, charts: dict, slides: dict):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

    print("\n📊 Data Processing:")
    print(f"   • {metrics['records_processed']} raw records ingested")
    print(f"   • {metrics['records_normalized']} policies normalized")
    print(f"   • {metrics['processing_time_ms']}ms processing time")
    print(f"   • {metrics.get('success_rate', 0):.1f}% success rate")

    print("\n📈 Visualizations Generated:")
    for name, path in charts.items():
        print(f"   • {name}: {path}")

    print("\n📑 Presentation Slides:")
    for name, path in slides.items():
        print(f"   • {name}: {path}")

    print("\n🚀 API Ready:")
    print("   Run 'python main.py --api' to start the server")
    print("   Documentation: http://localhost:8000/docs")

    print("\n" + "=" * 60)
    print("Project ready for MIT MBAn interview demonstration!")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EV Subsidy Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run full pipeline
  python main.py --data-only  # Generate and normalize data only
  python main.py --viz        # Generate visualizations only
  python main.py --api        # Start API server
        """
    )
    parser.add_argument("--data-only", action="store_true",
                       help="Run data pipeline only")
    parser.add_argument("--viz", action="store_true",
                       help="Generate visualizations only")
    parser.add_argument("--api", action="store_true",
                       help="Start API server")
    parser.add_argument("--no-slides", action="store_true",
                       help="Skip slide generation")

    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "=" * 60)
    print("EV SUBSIDY ANALYTICS PIPELINE")
    print("MIT MBAn Interview Project")
    print("=" * 60)

    try:
        if args.api:
            # Just start the API server
            run_api_server()
        elif args.data_only:
            # Run data pipeline only
            df, metrics = run_data_pipeline()
            print(f"\nData pipeline complete in {time.time() - start_time:.2f}s")
        elif args.viz:
            # Run visualizations only
            charts, slides = run_visualizations()
            print(f"\nVisualizations complete in {time.time() - start_time:.2f}s")
        else:
            # Run full pipeline
            df, metrics = run_data_pipeline()
            charts, slides = run_visualizations()
            print_summary(metrics, charts, slides)
            print(f"Total execution time: {time.time() - start_time:.2f}s")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
