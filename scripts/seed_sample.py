"""Seed the database with the sample conversation and run evaluation.

Usage:
    python -m scripts.seed_sample
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.session import init_db, get_db
from src.ingestion.normalizer import normalize_raw_transcript
from src.evaluation.pipeline import EvaluationPipeline


def main() -> None:
    """Load sample conversation, evaluate it, and print results."""
    print("=" * 70)
    print(" ConvoLens -- Sample Evaluation Demo")
    print("=" * 70)

    # Initialize database
    print("\n[1/4] Initializing database...")
    init_db()
    print("  [+] Database tables created")

    # Load sample conversation
    sample_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "samples"
        / "sample_conversation.json"
    )
    print(f"\n[2/4] Loading sample conversation from {sample_path.name}...")

    with open(sample_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    conversation = normalize_raw_transcript(raw_data)
    print(f"  [+] Loaded conversation: {conversation.conversation_id}")
    print(f"  [+] {len(conversation.turns)} turns, channel={conversation.channel}")
    print(f"  [+] Goal: {conversation.conversation_goal}")

    # Run evaluation
    print("\n[3/4] Running evaluation pipeline...")
    print("  (Using heuristic scoring -- set ANTHROPIC_API_KEY in .env for LLM judge)")

    with get_db() as db:
        pipeline = EvaluationPipeline(db=db)
        result = pipeline.evaluate(conversation)

    # Print results
    print(f"\n[4/4] Results for {result.conversation_id}")
    print("-" * 50)
    print(f"  Overall Composite Score: {result.overall_scores.composite:.2f}/5.00")
    print(f"    Accuracy:        {result.overall_scores.accuracy}/5")
    print(f"    Empathy:         {result.overall_scores.empathy}/5")
    print(f"    Flow:            {result.overall_scores.flow}/5")
    print(f"    Goal Completion: {result.overall_scores.goal_completion}/5")

    print(f"\n  Failures Detected: {len(result.conversation_failures)}")
    for i, f in enumerate(result.conversation_failures, 1):
        print(f"    {i}. [{f.severity.value}] {f.category.value} -- {f.subtype or 'general'}")
        if f.evidence_quote:
            quote = (
                f.evidence_quote[:100] + "..."
                if len(f.evidence_quote) > 100
                else f.evidence_quote
            )
            print(f"       Evidence: {quote}")

    print(f"\n  Root Causes: {len(result.root_causes)}")
    for rc in result.root_causes:
        print(f"    - {rc.root_cause.value} (confidence={rc.confidence:.0%})")
        if rc.recommended_fix:
            print(f"      Fix: {rc.recommended_fix[:100]}")

    if result.recommended_fixes:
        print("\n  Recommended Fixes:")
        for fix in result.recommended_fixes:
            print(f"    * {fix}")

    print("\n" + "=" * 70)
    print(" Evaluation saved to database. Start the server to query via API:")
    print("   uvicorn src.main:app --reload")
    print("=" * 70)


if __name__ == "__main__":
    main()
