# Changelog

All notable changes to the ConvoLens AI QA platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] - 2026-10-04

### Added
- **LLM-as-Judge Evaluation Engine**: Turn-by-turn and dialogue-level conversational audit powered by Claude 3.5 Sonnet and OpenAI GPT-4o with structured JSON outputs.
- **7-Layer Root-Cause Attribution Engine**: Automated diagnostic mapping of conversation failures into deterministic categories (`RC-PROMPT`, `RC-RAG-RETRIEVAL`, `RC-TOOL-SCHEMA`, `RC-CONTEXT-WINDOW`, `RC-GUARDRAIL`, `RC-KB-GAP`, `RC-POLICY-GAP`).
- **Zero-Trust PII Masking Pipeline**: Real-time regex and checksum-backed redactor for Indian enterprise data identifiers (Aadhaar, PAN, phone numbers, email addresses).
- **HDBSCAN Failure Clustering & Pattern Mining**: Unsupervised density-based clustering of semantic failures across call logs with automatic issue taxonomy generation.
- **FastAPI Production REST API**: 5 high-throughput endpoints (`/evaluate/conversation`, `/evaluate/turn`, `/results/{conversation_id}`, `/trends`, `/health`) with OpenAPI / Swagger documentation.
- **Interactive Vercel Dashboard**: Monochromatic, dark-themed responsive UI with real-time dialogue simulation, live streaming typing indicators, metric scoring bars, and RCA drilldown.
- **90% Test Coverage**: Comprehensive pytest test suite spanning 54 unit and integration tests (36 API route tests, 18 core evaluation engine tests).
- **Comprehensive Failure Taxonomy**: Multi-tier classification covering 7 L1 categories (`L1-HAL`, `L1-INS`, `L1-TOX`, `L1-POL`, `L1-REP`, `L1-CON`, `L1-LAT`) and 3 L2 speech layers (TTS, ASR, Turn-Taking).
- **Docker & Container Deployment**: Production multi-stage `Dockerfile` and `docker-compose.yml` for unified microservice orchestration.

### Performance
- Sub-100ms rule-based and PII redactor execution per turn.
- End-to-end evaluation turnaround: 2-5 seconds per multi-turn enterprise conversation.
