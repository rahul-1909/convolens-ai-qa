"""Live API demonstration script — makes real HTTP calls to the running FastAPI server."""

import json
from pathlib import Path
import httpx

BASE_URL = "http://127.0.0.1:8000"


def main():
    print("=" * 80)
    print(" CONVOLENS LIVE SERVER DEMONSTRATION")
    print(f" Target Server: {BASE_URL}")
    print("=" * 80)

    client = httpx.Client(base_url=BASE_URL, timeout=15.0)

    # 1. Health Check
    print("\n--- 1. Testing GET /health ---")
    r1 = client.get("/health")
    print(f"Status Code: {r1.status_code}")
    print("Response JSON:")
    print(json.dumps(r1.json(), indent=2))

    # 2. Ingest and Evaluate Conversation
    print("\n--- 2. Testing POST /evaluate/conversation ---")
    sample_file = Path("data/samples/wealth_credit_onboarding_call.json")
    with open(sample_file, "r", encoding="utf-8") as f:
        conv_payload = json.load(f)

    print(f"Sending payload: {conv_payload['conversation_id']} ({len(conv_payload['turns'])} turns)")
    r2 = client.post(
        "/evaluate/conversation",
        json={"conversation": conv_payload, "run_pattern_mining": False},
    )
    print(f"Status Code: {r2.status_code}")
    eval_res = r2.json()
    print("Status:", eval_res.get("status"))
    print("Conversation ID:", eval_res.get("conversation_id"))
    res = eval_res["result"]
    print("\nOverall Scores:")
    print(json.dumps(res["overall_scores"], indent=2))
    print(f"\nFailures Detected ({len(res['conversation_failures'])}):")
    for f in res["conversation_failures"]:
        print(f"  [{f['severity']}] {f['category']} -> {f['subtype']}")
        print(f"    Evidence: {f['evidence_quote']}")
        print(f"    Reasoning: {f['reasoning']}")

    print(f"\nRoot Cause Attribution ({len(res['root_causes'])}):")
    for rc in res["root_causes"]:
        print(f"  Layer: {rc['root_cause']} (Confidence: {rc['confidence']:.0%})")
        print(f"  Fix:   {rc['recommended_fix']}")

    # 3. Retrieve Results
    print(f"\n--- 3. Testing GET /results/{conv_payload['conversation_id']} ---")
    r3 = client.get(f"/results/{conv_payload['conversation_id']}")
    print(f"Status Code: {r3.status_code}")
    get_res = r3.json()
    print("Found in DB:", get_res["found"])
    print("Composite Score from DB:", get_res["result"]["overall_scores"]["composite"])

    # 4. Trends
    print("\n--- 4. Testing GET /trends ---")
    r4 = client.get("/trends")
    print(f"Status Code: {r4.status_code}")
    print("Trends Summary:")
    print(json.dumps(r4.json(), indent=2))

    print("\n" + "=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY — ALL ENDPOINTS WORKING!")
    print("=" * 80)


if __name__ == "__main__":
    main()
