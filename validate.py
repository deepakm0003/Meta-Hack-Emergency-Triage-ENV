#!/usr/bin/env python3
"""
Pre-submission validation script for Emergency Clinical Triage Environment.
Runs all checklist items and reports pass/fail for each.

Usage:
    # Start the server first:
    #   uvicorn main:app --host 0.0.0.0 --port 7860
    python validate.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import requests

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: Dict[str, str] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    results[name] = status
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Required files ─────────────────────────────────────────────────────────
section("1. Required Files")

required_files = [
    "inference.py", "main.py", "grader.py", "patient_generator.py",
    "outcome_simulator.py", "openenv.yaml", "Dockerfile", "requirements.txt",
    "README.md", "tasks/easy.py", "tasks/medium.py", "tasks/hard.py",
]
for f in required_files:
    check(f"File: {f}", os.path.exists(f))


# ── 2. inference.py in root ───────────────────────────────────────────────────
section("2. Inference Script")
check("inference.py is in root directory", os.path.exists("inference.py"))

# Check uses OpenAI client
with open("inference.py") as fh:
    inf_src = fh.read()
check("Uses 'from openai import OpenAI'", "from openai import OpenAI" in inf_src)
check("[START] log format present", "[START]" in inf_src)
check("[STEP] log format present", "[STEP]" in inf_src)
check("[END] log format present", "[END]" in inf_src)
check("Reads API_BASE_URL env var", "API_BASE_URL" in inf_src)
check("Reads MODEL_NAME env var", "MODEL_NAME" in inf_src)
check("Reads HF_TOKEN env var", "HF_TOKEN" in inf_src)
check("No hardcoded API key", "gsk_" not in inf_src and "sk-" not in inf_src.replace("sk-placeholder", ""))


# ── 3. openenv.yaml ───────────────────────────────────────────────────────────
section("3. OpenEnv Spec (openenv.yaml)")
try:
    import yaml  # type: ignore
    with open("openenv.yaml") as fh:
        spec = yaml.safe_load(fh)
    check("openenv.yaml parses", True)
    check("Has 'name' field", "name" in spec)
    check("Has 'version' field", "version" in spec)
    check("Has 'tasks' list", "tasks" in spec and len(spec["tasks"]) >= 3)
    check("Has 3+ task definitions", len(spec.get("tasks", [])) >= 3)
    check("API endpoints defined", "api" in spec)
    check("env_vars documented", "env_vars" in spec)
    task_ids = [t["id"] for t in spec.get("tasks", [])]
    check("static_triage task defined", "static_triage" in task_ids)
    check("late_info_triage task defined", "late_info_triage" in task_ids)
    check("mass_casualty task defined", "mass_casualty" in task_ids)
except ImportError:
    with open("openenv.yaml") as fh:
        raw = fh.read()
    check("openenv.yaml exists and readable", True, "yaml lib not installed — skipping parse")
    check("Has tasks section", "tasks:" in raw)
    check("Has static_triage", "static_triage" in raw)
    check("Has late_info_triage", "late_info_triage" in raw)
    check("Has mass_casualty", "mass_casualty" in raw)


# ── 4. Dockerfile ─────────────────────────────────────────────────────────────
section("4. Dockerfile")
with open("Dockerfile") as fh:
    df = fh.read()
check("Dockerfile exists", True)
check("EXPOSE 7860", "EXPOSE 7860" in df)
check("CMD runs uvicorn", "uvicorn" in df and "main:app" in df)
check("Copies requirements.txt", "requirements.txt" in df)
check("pip install in Dockerfile", "pip install" in df)


# ── 5. requirements.txt ───────────────────────────────────────────────────────
section("5. requirements.txt")
with open("requirements.txt") as fh:
    reqs = fh.read().lower()
check("fastapi listed", "fastapi" in reqs)
check("uvicorn listed", "uvicorn" in reqs)
check("scipy listed", "scipy" in reqs)
check("openai listed", "openai" in reqs)
check("requests listed", "requests" in reqs)


# ── 6. Live server checks ─────────────────────────────────────────────────────
section("6. Live Server (must be running)")

def get(path: str, timeout: int = 10) -> Any:
    return requests.get(f"{ENV_BASE_URL}{path}", timeout=timeout).json()

def post(path: str, body: dict, timeout: int = 10) -> Any:
    return requests.post(f"{ENV_BASE_URL}{path}", json=body, timeout=timeout).json()

# Health check
try:
    h = get("/health")
    check("GET /health returns 200", h.get("status") == "ok", str(h.get("status")))
except Exception as e:
    check("GET /health returns 200", False, str(e))
    print("\n  ⚠️  Server not running! Start it with:")
    print("     uvicorn main:app --host 0.0.0.0 --port 7860")
    print("\n  Skipping remaining live checks.\n")
    sys.exit(_summarize())

# Tasks endpoint
try:
    t = get("/tasks")
    check("GET /tasks returns task list", "tasks" in t)
except Exception as e:
    check("GET /tasks returns task list", False, str(e))

# Test all 3 tasks
section("7. Task Reset + Step Validation")
for task_id in ["static_triage", "late_info_triage", "mass_casualty"]:
    try:
        # reset
        r = post("/reset", {"task_id": task_id})
        ep_id = r.get("episode_id")
        obs   = r.get("observation", {})
        patients = obs.get("patients", [])

        check(f"{task_id}: /reset returns episode_id", bool(ep_id))
        check(f"{task_id}: /reset returns patients", len(patients) > 0,
              f"{len(patients)} patients")

        # one step
        ranking = [p["id"] for p in patients]
        step_body = {
            "episode_id": ep_id,
            "action": {
                "priority_ranking": ranking,
                "reasoning": (
                    "ESI triage: critical patients assessed. SpO2, GCS, HR, troponin, "
                    "sepsis, STEMI, stroke indicators reviewed. Hemodynamic stability "
                    "evaluated. Airway compromise risk assessed. Deterioration risk high "
                    "for ESI-1 patients."
                ),
                "allocations": {} if task_id == "mass_casualty" else None,
            }
        }
        s = post("/step", step_body)
        reward = s.get("reward", -1)
        breakdown = s.get("info", {}).get("reward_breakdown", {})

        check(f"{task_id}: /step returns reward in [0,1]",
              0.0 <= reward <= 1.0, f"reward={reward}")
        check(f"{task_id}: reward_breakdown present",
              bool(breakdown), str(list(breakdown.keys())[:3]))
        check(f"{task_id}: ranking_score in breakdown",
              "ranking_score" in breakdown)
        check(f"{task_id}: all breakdown values in [0,1]",
              all(0.0 <= v <= 1.0 for v in breakdown.values() if v is not None))

    except Exception as e:
        check(f"{task_id}: reset+step", False, str(e)[:80])

    time.sleep(1)


# ── Summary ───────────────────────────────────────────────────────────────────
def _summarize() -> int:
    passed = sum(1 for v in results.values() if v == PASS)
    failed = sum(1 for v in results.values() if v == FAIL)
    total  = len(results)
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    if failed > 0:
        print("\n  Failed checks:")
        for name, status in results.items():
            if status == FAIL:
                print(f"    ❌  {name}")
    print()
    return 1 if failed > 0 else 0


sys.exit(_summarize())
