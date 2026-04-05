# 🏥 Emergency Clinical Triage Environment

An **OpenEnv-compatible** reinforcement learning environment where an LLM agent acts as an ER triage nurse. The agent must correctly prioritize patients by medical urgency (ESI 1–5), update its beliefs when new lab results arrive mid-episode, and—in the hardest task—allocate scarce ICU beds during a mass casualty event.

---

## Hugging Face URL :- https://huggingface.co/spaces/deepak0003/Meta-Hack

## 🏆 Benchmark Results

| Task | Score | Status |
|---|---|---|
| `static_triage` | **0.90** | ✅ Pass |
| `late_info_triage` | **0.85** | ✅ Pass |
| `mass_casualty` | **0.83** | ✅ Pass |
| **Average** | **0.8320** | ✅ |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the environment server
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 3. Run inference
```bash
export API_BASE_URL=https://api.groq.com/openai/v1   # or any OpenAI-compatible endpoint
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_api_key_here
python inference.py
```

**Windows:** Edit `run_inference.bat` with your API key and double-click to run.

### Docker
```bash
docker build -t emergency-triage-env .
docker run -p 7860:7860 emergency-triage-env
```

---

## 📋 Tasks

| Task ID | Difficulty | Patients | Description |
|---|---|---|---|
| `static_triage` | Easy | 5 | Complete vitals, rank by ESI priority |
| `late_info_triage` | Medium | 6 | Labs hidden; injected at step 3 |
| `mass_casualty` | Hard | 12 | Conflicting notes, ICU constraints, outcome simulation |

---

## 🔌 API Reference

### `POST /reset`
```json
{ "task_id": "static_triage" }
```

### `POST /step`
```json
{
  "episode_id": "uuid",
  "action": {
    "priority_ranking": ["P003", "P001", "P005"],
    "reasoning": "P003 shows STEMI signs with troponin 2.4 ng/mL...",
    "allocations": { "P003": "ICU", "P001": "resus" }
  }
}
```

### `GET /state?episode_id=<uuid>`

Returns current episode observation with patient list, step number, and any new lab info.

---

## 📐 Action Space

| Field | Type | Required | Description |
|---|---|---|---|
| `priority_ranking` | `List[str]` | ✅ Always | Patient IDs, most critical first |
| `reasoning` | `str` | ✅ Always | Clinical justification (80+ words recommended) |
| `allocations` | `Dict[str,str]` | Hard task only | `"ICU"` \| `"resus"` \| `"general"` |

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `patients` | `List[Patient]` | Current patient list with vitals |
| `new_info` | `str` | Lab results injected mid-episode (empty if none) |
| `step` | `int` | Current step number |
| `max_steps` | `int` | Episode length (always 5) |
| `icu_beds_available` | `int` | Hard task only — bed capacity |

### Patient Object
| Field | Description |
|---|---|
| `id` | Patient identifier (e.g. `P001`) |
| `age` | Patient age |
| `chief_complaint` | Primary presenting complaint |
| `vitals` | `bp`, `hr`, `rr`, `spo2`, `temp`, `gcs` |
| `nurse_note` | Triage nurse observations |
| `labs` | Lab results (`null` until injected at step 3) |

---

## 🏆 Reward Function

| Component | Max | Condition |
|---|---|---|
| `ranking_score` | +0.40 | Kendall τ vs ground-truth ESI order |
| `belief_update_score` | +0.25 | ESI-1 patient moves UP after lab injection |
| `reasoning_score` | +0.20 | Clinical keywords in reasoning string |
| `capacity_score` | +0.10 | ICU allocations ≤ available beds (hard only) |
| `consistency_bonus` | +0.05 | Top-3 unchanged across last 3 steps |
| `critical_miss_penalty` | up to −0.25 | Any ESI-1 patient in bottom 40% of ranking |

**Final reward** = sum of all components, clamped to `[0.0, 1.0]`

> ⚠️ The critical miss penalty creates a non-linear cliff: agents that correctly identify all life-threatening patients score ~0.65+, while agents that miss them score near 0.0.

---

## 🩺 Clinical Scenarios

10 hardcoded archetypes spanning all ESI levels:

| Scenario | ESI | Key Features |
|---|---|---|
| STEMI | 1 | Troponin HIGH, ST elevation, diaphoresis |
| Stroke | 1 | FAST+ exam, BP 195/105, GCS 13 |
| Sepsis | 1 | Lactate 4.2, HR 126, temp 39.8 |
| Appendicitis | 2 | Rebound tenderness, WBC HIGH |
| Asthma Attack | 2 | SpO2 90%, accessory muscles, pCO2 HIGH |
| DKA | 2 | Glucose 498, pH 7.21, Kussmaul resp. |
| Head Trauma | 2 | GCS 14, epidural hematoma |
| UTI | 4 | Nitrites+, no systemic symptoms |
| Panic Attack | 4 | Troponin negative, normal ECG |
| Sprained Ankle | 5 | Ottawa rules negative |

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | OpenAI-compatible API endpoint (default: Groq) |
| `MODEL_NAME` | Model identifier (e.g. `llama-3.3-70b-versatile`, `gpt-4o`) |
| `HF_TOKEN` | API key for the LLM provider |
| `ENV_BASE_URL` | Base URL of this environment server (default: `http://localhost:7860`) |

---

## 📊 Scoring Interpretation

| Score Range | Interpretation |
|---|---|
| 0.82 – 1.00 | Expert triage — all critical patients correctly identified |
| 0.65 – 0.84 | Competent — minor ordering errors |
| 0.35 – 0.64 | Borderline — misses some urgency signals |
| 0.00 – 0.34 | Critical failure — missed ESI-1 patient (penalty applied) |

---

## 🏗️ Architecture

```
POST /reset  →  builds episode from task config (patient generator)
POST /step   →  scores action, injects labs at step 3, runs outcome sim on final step
GET  /state  →  returns current episode state
GET  /health →  health check
GET  /tasks  →  lists available task IDs
```

Built for **Hugging Face Spaces** (port 7860) and compatible with any OpenEnv-compliant runner.

