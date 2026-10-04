# ConvoLens Development & API Guide

## 1. Quickstart & Local Setup

### Prerequisites
- Python 3.11+
- Virtualenv or Poetry
- SQLite (built-in default) or PostgreSQL 15+

### Installation

```bash
# Clone and enter the repository
cd convolens-ai-qa

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key variables to review:
```ini
DATABASE_URL=sqlite:///./convolens.db
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your-anthropic-key-here
LLM_MODEL=claude-sonnet-4-20250514
PII_MASKING_ENABLED=true
```
*(Note: If no API key is provided, ConvoLens operates in deterministic heuristic fallback mode without breaking).*

---

## 2. Running the Application

### Seed Sample Data & Run End-to-End Evaluation
Run the standalone demonstration script to initialize the database, parse a real BFSI loan-collection dialogue, evaluate all 15 turns, and output the report:

```bash
python -m scripts.seed_sample
```

### Start the FastAPI Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation is available at:
`http://localhost:8000/docs`

---

## 3. API Examples

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "database": "connected",
  "timestamp": "2026-10-04T09:45:00.000000"
}
```

---

### Evaluate Conversation (Structured Payload)
```bash
curl -X POST "http://localhost:8000/evaluate/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {
      "conversation_id": "conv-test-101",
      "customer_id": "CUST-4521",
      "agent_version": "v2.4.0",
      "conversation_goal": "loan_collection",
      "channel": "voice",
      "turns": [
        {
          "turn_id": "t-1",
          "turn_index": 0,
          "speaker": "agent",
          "transcript": "Hello, this is ABC Finance regarding your pending EMI payment of Rs. 15,000.",
          "asr_confidence": 0.98,
          "kb_snippets": ["Overdue EMI is Rs. 15,000 due Sept 25"]
        },
        {
          "turn_id": "t-2",
          "turn_index": 1,
          "speaker": "customer",
          "transcript": "I lost my job last week, can I pay less?",
          "asr_confidence": 0.91
        },
        {
          "turn_id": "t-3",
          "turn_index": 2,
          "speaker": "agent",
          "transcript": "I understand. As per our records you have a special discount to pay only Rs. 5,000.",
          "asr_confidence": 0.95
        }
      ]
    },
    "run_pattern_mining": false
  }'
```

---

### Retrieve Evaluation Results
```bash
curl -X GET "http://localhost:8000/results/conv-test-101"
```

---

### Retrieve Quality Trends & Failure Distributions
```bash
curl -X GET "http://localhost:8000/trends"
```

**Response:**
```json
{
  "total_conversations": 1,
  "total_failures": 2,
  "average_composite_score": 3.45,
  "failure_distribution": [
    {
      "category": "L1-HAL",
      "severity": "S2",
      "count": 1,
      "percentage": 50.0
    },
    {
      "category": "L2-ASR",
      "severity": "S3",
      "count": 1,
      "percentage": 50.0
    }
  ],
  "top_root_causes": [
    {
      "root_cause": "RC-KB",
      "count": 1,
      "avg_confidence": 0.75
    },
    {
      "root_cause": "RC-ASR",
      "count": 1,
      "avg_confidence": 0.9
    }
  ]
}
```

---

## 4. Running Tests

Run the test suite with pytest:

```bash
pytest -v
```

To run with coverage reporting:
```bash
pytest --cov=src -v
```

---

## 5. Docker Deployment

### Run API & PostgreSQL with Docker Compose
```bash
docker-compose up --build -d
```

### Stop Services
```bash
docker-compose down
```
