"""
Entry point for the misinformation explanation pipeline.

Usage examples:

  # Run both datasets with Gemini 2.0 Flash (default)
  python run_misinfo.py --datasets both

  # Run only the INTEL dataset with a specific model
  python run_misinfo.py --datasets intel --model gemini-2.5-pro

  # Run PUBHEALTH with OpenAI
  python run_misinfo.py --datasets pubhealth --model gpt-4o-mini-2024-07-18

  # Override API key inline
  python run_misinfo.py --datasets both --api-key YOUR_KEY

  # Disable resume (reprocess everything from scratch)
  python run_misinfo.py --datasets both --no-resume
"""

import argparse
import os
import sys

# Allow importing from this directory when run directly
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_misinfo import LLMClient, process_csv

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.0-flash"

INTEL_INPUT    = os.path.join(os.path.dirname(__file__), "intel_misclassified_top200.csv")
PUBHEALTH_INPUT = os.path.join(os.path.dirname(__file__), "pubhealth_misclassified_top200.csv")

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "misinfo_explanations")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM explanations for BERT misclassified misinformation examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        choices=["intel", "pubhealth", "both"],
        default="both",
        help="Which dataset(s) to process (default: both)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (falls back to GOOGLE_API_KEY / OPENAI_API_KEY env vars)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write output CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess all rows even if output CSV already exists",
    )
    parser.add_argument(
        "--intel-input",
        default=INTEL_INPUT,
        help=f"Path to INTEL input CSV (default: {INTEL_INPUT})",
    )
    parser.add_argument(
        "--pubhealth-input",
        default=PUBHEALTH_INPUT,
        help=f"Path to PUBHEALTH input CSV (default: {PUBHEALTH_INPUT})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Misinformation Explanation Pipeline")
    print("=" * 70)
    print(f"  Model      : {args.model}")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Resume     : {not args.no_resume}")
    print("=" * 70)

    # Initialise the LLM client once (shared across both datasets)
    client = LLMClient(model=args.model, api_key=args.api_key)

    tasks = []
    if args.datasets in ("intel", "both"):
        tasks.append({
            "name": "INTEL",
            "dataset": "intel",
            "input": args.intel_input,
            "output": os.path.join(
                args.output_dir,
                f"intel_misclassified_top200_explained.csv",
            ),
        })
    if args.datasets in ("pubhealth", "both"):
        tasks.append({
            "name": "PUBHEALTH",
            "dataset": "pubhealth",
            "input": args.pubhealth_input,
            "output": os.path.join(
                args.output_dir,
                f"pubhealth_misclassified_top200_explained.csv",
            ),
        })

    for task in tasks:
        print(f"\n{'─'*70}")
        print(f"Processing: {task['name']}")
        print(f"  Input : {task['input']}")
        print(f"  Output: {task['output']}")
        print(f"{'─'*70}")

        process_csv(
            input_path=task["input"],
            output_path=task["output"],
            dataset=task["dataset"],
            client=client,
            temperature=args.temperature,
            resume=not args.no_resume,
        )

    print("\n" + "=" * 70)
    print("All tasks complete.")
    print(f"Final cost summary: {client.cost_tracker.summary()}")
    print("=" * 70)


if __name__ == "__main__":
    main()