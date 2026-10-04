# ConvoLens Failure Taxonomy & Scoring Rubric

## 1. Overview
ConvoLens implements a two-level failure taxonomy designed specifically for enterprise voice and chat AI agents in regulated domains such as BFSI (Banking, Financial Services, and Insurance), collections, customer support, and sales onboarding.

---

## 2. Two-Level Failure Taxonomy

### Level 1: Conversational & Cognitive Failures

| Code | Category | Default Severity | Description | Subtypes |
| :--- | :--- | :---: | :--- | :--- |
| `L1-HAL` | **Hallucination** | **S1** | Agent fabricates facts, calculations, rates, or policies not grounded in KB or tool outputs. | `factual_fabrication`, `policy_misstatement`, `capability_hallucination`, `numerical_error` |
| `L1-CTX` | **Context Loss** | **S2** | Agent forgets, contradicts, or misattributes information provided earlier in the dialogue. | `forgotten_slot`, `contradicted_self`, `ignored_correction`, `wrong_reference` |
| `L1-LOOP` | **Loops / Repetition** | **S2** | Agent repeats questions, statements, or tool calls without moving the conversation forward. | `question_repeat`, `phrase_repeat`, `action_loop`, `stuck_state` |
| `L1-INT` | **Intent Misread** | **S2** | Agent misunderstands the user's explicit or implicit intent, directing to wrong branch. | `wrong_intent_classification`, `partial_intent_capture`, `premature_routing`, `missed_implicit_intent` |
| `L1-OBJ` | **Weak Objection Handling** | **S3** | Agent ignores customer hesitations or delivers generic, dismissive responses. | `ignored_objection`, `generic_response`, `premature_escalation`, `missed_buying_signal` |
| `L1-INS` | **Instruction Violation** | **S1** | Agent breaches mandatory script guidelines, disclosures, tone constraints, or security rules. | `skipped_disclosure`, `wrong_script_order`, `unauthorized_promise`, `tone_violation` |
| `L1-GOAL` | **Goal Failure** | **S1** | Conversation terminates without completing the required business objective. | `incomplete_collection`, `false_completion`, `premature_termination`, `wrong_outcome` |

### Level 2: Speech-Layer Failures

| Code | Category | Default Severity | Description | Subtypes |
| :--- | :--- | :---: | :--- | :--- |
| `L2-ASR` | **ASR Error** | **S3** | Speech-to-text misinterprets audio (homophones, names, digits, background noise). | `homophone_confusion`, `name_garbling`, `number_error`, `language_switch_miss` |
| `L2-TTS` | **TTS Mispronunciation** | **S3** | Text-to-speech output garbles critical proper nouns, amounts, or acronyms. | `name_mispronunciation`, `number_mispronunciation`, `acronym_error`, `language_mismatch` |
| `L2-BARG` | **Barge-in / Interruption** | **S3** | Turn-taking and Voice Activity Detection (VAD) failures, talking over user or cutting off. | `ignored_bargein`, `premature_cutoff`, `silence_timeout`, `overlap_confusion` |

---

## 3. Severity Scale (S1–S4)

* **S1 — Critical**: Immediate conversation failure, legal/regulatory compliance breach, or material customer harm (e.g. fabricated interest rate, skipped mandatory disclosures).
* **S2 — Major**: Goal completion severely jeopardized, loop requires manual intervention, or critical slot forgotten.
* **S3 — Moderate**: Degraded customer experience, conversational friction, or minor speech-layer flaw, but dialogue recovers.
* **S4 — Minor**: Cosmetic issue, minor acoustic artifact, or awkward phrasing with no factual or functional impact.

---

## 4. 1–5 Scoring Rubric with Behavioral Anchors

| Dimension | Weight | Score 5 (Exceptional) | Score 3 (Marginal / Pass) | Score 1 (Critical Failure) |
| :--- | :---: | :--- | :--- | :--- |
| **Accuracy** | 35% | 100% grounded in KB/tool output; zero hallucinations or math errors. | Generally correct, but one minor unverified claim or ambiguous figure. | Fabricates core facts; gives misleading numbers or false policies. |
| **Empathy** | 20% | Attuned to customer emotion; acknowledges difficulties warmly and professionally. | Functional and polite, but mechanical tone with no active acknowledgment of cues. | Aggressive, rude, robotic, or dismissive when user expresses hardship. |
| **Flow** | 20% | Flawless turn-taking, seamless transitions, concise progress. | Minor friction or one redundant question; slight awkward pause. | Infinite loop, incoherent turn ordering, or repetitive interrogation. |
| **Goal Completion** | 25% | All primary and secondary objectives achieved; verification complete. | Core goal partially satisfied, but follow-up steps or confirmations left vague. | Conversation dropped, wrong resolution reached, or zero progress made. |

**Composite Formula**:
$$\text{Composite} = 0.35 \times \text{Accuracy} + 0.20 \times \text{Empathy} + 0.20 \times \text{Flow} + 0.25 \times \text{Goal Completion}$$

---

## 5. Root-Cause Attribution Layers

1. **`RC-PROMPT` (Prompt / Instructions)**: Ambiguous system prompt, missing few-shot examples, or conflicting guidelines.
2. **`RC-KB` (Knowledge Base Gap)**: The needed fact was absent, outdated, or poorly retrieved from the vector store.
3. **`RC-TOOL` (Tool / API Failure)**: External service timed out, returned non-200 error, or payload format mismatch.
4. **`RC-CONTEXT` (Context Management)**: Context window truncation, missing memory retention, or state tracking error.
5. **`RC-ASR` (ASR / Audio Ingestion)**: Acoustic degradation, low transcription confidence, or mistranscribed entities.
6. **`RC-TTS` (TTS / Synthesis)**: Synthesis engine mispronounced domain terminology or acronym.
7. **`RC-WORKFLOW` (Workflow Orchestration)**: State-machine routing error, flawed escalation logic, or premature termination.
