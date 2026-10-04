"""Enterprise Showcase Demo — ConvoLens Automated Quality Evaluation.

Evaluates multi-turn realistic enterprise dialogues (BFSI Collections,
Wealth Card Onboarding) with real failure modes, PII masking, and
root-cause layer attribution.

Usage:
    python -m scripts.demo_showcase
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
from src.pii.masker import get_masker


def print_score_bar(label: str, score: float, max_score: int = 5) -> None:
    """Print an ASCII bar chart representing rubric score."""
    filled = int(round((score / max_score) * 20))
    bar = "#" * filled + "-" * (20 - filled)
    print(f"    {label:<18} [{bar}] {score:.2f} / {max_score}.00")


def run_scenario(sample_filename: str, title: str, description: str) -> None:
    """Load, evaluate, and display detailed results for a scenario."""
    print("\n" + "=" * 80)
    print(f" SCENARIO: {title.upper()}")
    print("=" * 80)
    print(f"Context: {description}\n")

    sample_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "samples"
        / sample_filename
    )

    with open(sample_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    conversation = normalize_raw_transcript(raw_data)
    masker = get_masker(enabled=True)

    print(f"[*] Ingested Conversation: {conversation.conversation_id}")
    print(f"    Agent Version: {conversation.agent_version} | Goal: {conversation.conversation_goal}")
    print(f"    Channel: {conversation.channel} | Language: {conversation.language} | Turns: {len(conversation.turns)}\n")

    print("[-] Sample Dialogue Turns (showing PII Masking):")
    for t in conversation.turns[:4]:
        masked_text = masker.mask(t.transcript)
        speaker_tag = "[AGENT]" if t.speaker.value == "agent" else "[USER] "
        print(f"    Turn {t.turn_index:02d} {speaker_tag}: {masked_text}")
    if len(conversation.turns) > 4:
        print(f"    ... [{len(conversation.turns) - 4} additional turns evaluated]")

    # Run evaluation
    with get_db() as db:
        pipeline = EvaluationPipeline(db=db)
        result = pipeline.evaluate(conversation)

    print("\n[+] Quality Scoring Rubric Breakdown:")
    print_score_bar("Accuracy (35%)", result.overall_scores.accuracy)
    print_score_bar("Empathy (20%)", result.overall_scores.empathy)
    print_score_bar("Flow (20%)", result.overall_scores.flow)
    print_score_bar("Goal Comp. (25%)", result.overall_scores.goal_completion)
    print("-" * 50)
    print_score_bar("COMPOSITE SCORE", result.overall_scores.composite)

    print(f"\n[!] Detected Quality Failures ({len(result.conversation_failures)} found):")
    if not result.conversation_failures:
        print("    No rule-based or heuristic anomalies detected.")
    else:
        for i, f in enumerate(result.conversation_failures, 1):
            print(f"    {i}. [{f.severity.value}] {f.category.value} -> {f.subtype or 'general'}")
            if f.evidence_quote:
                print(f"       Evidence: \"{f.evidence_quote.strip()}\"")
            if f.reasoning:
                print(f"       Reasoning: {f.reasoning.strip()}")

    print(f"\n[*] Root-Cause Layer Attribution ({len(result.root_causes)} signals):")
    for rc in result.root_causes:
        print(f"    - Layer: {rc.root_cause.value:<12} (Confidence: {rc.confidence:.0%})")
        if rc.signals:
            for s in rc.signals:
                print(f"      Signal: {s}")
        if rc.recommended_fix:
            print(f"      Fix:    {rc.recommended_fix}")


def main() -> None:
    print("=" * 80)
    print(" CONVOLENS -- Automated Quality Evaluation & Failure Taxonomy Showcase")
    print("=" * 80)

    # Initialize Database
    init_db()

    # Scenario 1: BFSI Loan Collection & Calamity Hardship Call
    run_scenario(
        sample_filename="bfsi_hardship_collection_call.json",
        title="BFSI Hardship Collection & Unauthorized Moratorium Waiver",
        description="Retail borrower affected by floods. Agent hallucinates zero interest moratorium (L1-HAL), acoustic homophone error on token payment (L2-ASR).",
    )

    # Scenario 2: Wealth Credit Card Onboarding & Compliance Slip
    run_scenario(
        sample_filename="wealth_credit_onboarding_call.json",
        title="Wealth Credit Card Sales & Mandatory Disclosure Breach",
        description="Premium credit card sales. Agent skips mandatory RBI annual fee disclosure (L1-INS) and deflects user privacy objection (L1-OBJ). Demonstrates PII masking.",
    )

    print("\n" + "=" * 80)
    print(" ALL SCENARIOS EVALUATED & PERSISTED.")
    print(" Query analytics, trends, and failure clusters via API:")
    print("   curl http://localhost:8000/trends")
    print("=" * 80)


if __name__ == "__main__":
    main()
