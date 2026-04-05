"""
Emergency Clinical Triage Environment — FastAPI Application
Endpoints: POST /reset  POST /step  GET /state  GET /health  GET /tasks
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from patient_generator import get_public_patient, inject_labs
from grader import compute_reward
from outcome_simulator import simulate_outcomes
from tasks import TASK_BUILDERS, TASK_METADATA

app = FastAPI(
    title="Emergency Clinical Triage Environment",
    description="OpenEnv-compatible ER triage environment for LLM agent evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EPISODES: Dict[str, Dict[str, Any]] = {}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "static_triage"   # default: ping with {} still returns 200


class Action(BaseModel):
    priority_ranking: List[str]
    reasoning: str = ""
    allocations: Optional[Dict[str, str]] = None


class StepRequest(BaseModel):
    episode_id: str
    action: Action


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_observation(episode: Dict[str, Any], new_info: str = "") -> Dict[str, Any]:
    include_conflicting = episode["task_id"] in ("late_info_triage", "mass_casualty")
    public_patients = [
        get_public_patient(p, include_conflicting=include_conflicting)
        for p in episode["patients"]
    ]
    obs: Dict[str, Any] = {
        "patients": public_patients,
        "new_info": new_info,
        "step": episode["step"],
        "max_steps": episode["max_steps"],
    }
    if episode["task_id"] == "mass_casualty":
        obs["icu_beds_available"] = episode["icu_beds_available"]
        obs["resus_bays_available"] = episode["resus_bays_available"]
        if episode.get("resources"):
            obs["resources"] = episode["resources"]
    return obs


def _generate_new_info(patients, step: int, inject_step: Optional[int]) -> str:
    if inject_step is None or step != inject_step:
        return ""
    lines = []
    for p in patients:
        if p.get("labs"):
            lab_str = ", ".join(f"{k}: {v}" for k, v in p["labs"].items())
            lines.append(f"Lab results for {p['id']} ({p.get('_scenario', '')}) -> {lab_str}")
    return "\n".join(lines) if lines else ""


# ── POST /reset ───────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(body: Optional[ResetRequest] = None) -> Dict[str, Any]:
    if body is None:
        body = ResetRequest()
    if body.task_id not in TASK_BUILDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{body.task_id}'. Valid: {list(TASK_BUILDERS.keys())}",
        )

    cfg = TASK_BUILDERS[body.task_id]()
    episode_id = str(uuid.uuid4())

    episode: Dict[str, Any] = {
        "episode_id": episode_id,
        "task_id": body.task_id,
        "patients": cfg["patients"],
        "step": 0,
        "max_steps": cfg["max_steps"],
        "icu_beds_available": cfg.get("icu_beds_available") or 3,
        "resus_bays_available": cfg.get("resus_bays_available") or 2,
        "inject_labs_at_step": cfg.get("inject_labs_at_step"),
        "resources": cfg.get("resources"),
        "done": False,
        "best_reward": 0.0,
        "reward_history": [],
        "ranking_history": [],
    }
    EPISODES[episode_id] = episode

    return {
        "episode_id": episode_id,
        "task_id": body.task_id,
        "observation": _build_observation(episode),
        "done": False,
        "reward": 0.0,
    }


# ── POST /step ────────────────────────────────────────────────────────────────

@app.post("/step")
def step(body: StepRequest) -> Dict[str, Any]:
    if body.episode_id not in EPISODES:
        raise HTTPException(status_code=404, detail="Episode not found. Call /reset first.")

    episode = EPISODES[body.episode_id]
    if episode["done"]:
        raise HTTPException(status_code=400, detail="Episode already done. Call /reset.")

    episode["step"] += 1
    current_step = episode["step"]
    inject_step  = episode["inject_labs_at_step"]

    # Inject labs BEFORE scoring so grader sees up-to-date lab values
    new_info = ""
    if inject_step and current_step == inject_step:
        episode["patients"] = inject_labs(episode["patients"])
        new_info = _generate_new_info(episode["patients"], current_step, inject_step)

    ranking = list(body.action.priority_ranking)   # ensure plain list copy

    # Belief update window: step just before injection vs step just after
    ranking_before: Optional[List[str]] = None
    ranking_after:  Optional[List[str]] = None
    if inject_step:
        history = episode["ranking_history"]
        before_idx = inject_step - 2   # 0-based: step (inject_step-1) submitted ranking
        after_idx  = inject_step       # 0-based: step (inject_step+1) submitted ranking
        if 0 <= before_idx < len(history):
            ranking_before = history[before_idx]
        # after_idx points to the ranking we're about to append — check after append below

    # Append ranking history BEFORE calling grader so consistency_bonus sees it
    episode["ranking_history"].append(ranking)

    # Now resolve after-ranking (current step if it is inject_step+1)
    if inject_step and after_idx < len(episode["ranking_history"]):
        ranking_after = episode["ranking_history"][after_idx]

    # Outcome simulation for hard task on final step
    outcome_score:  Optional[float] = None
    outcome_detail: Optional[Dict]  = None
    if episode["task_id"] == "mass_casualty" and current_step >= episode["max_steps"]:
        sim = simulate_outcomes(
            agent_ranking=ranking,
            patients=episode["patients"],
            allocations=body.action.allocations,
        )
        outcome_score  = sim["outcome_score"]
        outcome_detail = sim

    grade = compute_reward(
        agent_ranking=ranking,
        reasoning=body.action.reasoning,
        patients=episode["patients"],
        task_id=episode["task_id"],
        allocations=body.action.allocations,
        icu_beds_available=episode["icu_beds_available"],
        resus_bays_available=episode["resus_bays_available"],
        ranking_before=ranking_before,
        ranking_after=ranking_after,
        outcome_score=outcome_score,
        ranking_history=episode["ranking_history"],
    )

    raw_reward = grade["reward"]

    # ALWAYS protect best reward — clamp immediately, before done check
    episode["best_reward"] = max(episode["best_reward"], raw_reward)
    episode["reward_history"].append(raw_reward)

    done = current_step >= episode["max_steps"]
    episode["done"] = done

    response: Dict[str, Any] = {
        "observation": _build_observation(episode, new_info=new_info),
        "reward": raw_reward,
        "done": done,
        "info": {
            "reward_breakdown": grade["reward_breakdown"],
            "step": current_step,
            "best_reward_so_far": episode["best_reward"],
        },
    }
    if outcome_detail:
        response["info"]["outcome_simulation"] = outcome_detail

    return response


# ── GET /state ────────────────────────────────────────────────────────────────

@app.get("/state")
def state(episode_id: str) -> Dict[str, Any]:
    if episode_id not in EPISODES:
        raise HTTPException(status_code=404, detail="Episode not found.")
    ep = EPISODES[episode_id]
    return {
        "episode_id": episode_id,
        "task_id": ep["task_id"],
        "observation": _build_observation(ep),
        "done": ep["done"],
        "best_reward": ep["best_reward"],
        "reward_history": ep["reward_history"],
        "step": ep["step"],
        "max_steps": ep["max_steps"],
    }


# ── GET /score/{episode_id} ───────────────────────────────────────────────────

@app.get("/score/{episode_id}")
def score(episode_id: str) -> Dict[str, Any]:
    """Return the best reward seen across all steps for a given episode."""
    if episode_id not in EPISODES:
        raise HTTPException(status_code=404, detail="Episode not found.")
    ep = EPISODES[episode_id]
    return {
        "episode_id": episode_id,
        "score": ep["best_reward"],
        "reward_history": ep["reward_history"],
        "done": ep["done"],
    }


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "env": "emergency-triage-env",
        "version": "1.0.0",
        "tasks": list(TASK_BUILDERS.keys()),
    }


# ── GET /tasks ────────────────────────────────────────────────────────────────

@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "tasks": TASK_METADATA,
        "api": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health",
        },
        "reward_components": {
            "ranking_score": {"max": 0.40, "description": "Kendall tau + weighted position score vs ground-truth ESI order"},
            "belief_update_score": {"max": 0.25, "description": "Fraction of critical lab patients correctly moved up after injection"},
            "reasoning_score": {"max": 0.20, "description": "Clinical keyword density + reasoning length"},
            "capacity_score": {"max": 0.10, "description": "Hard task only: correct ICU/resus allocation within bed limits"},
            "consistency_bonus": {"max": 0.05, "description": "Top-3 ranking stable across last 3 steps"},
            "critical_miss_penalty": {"max": -0.40, "description": "Tiered penalty for misplacing ESI-1/2 patients"},
        },
    }


# ── GET / (root) ──────────────────────────────────────────────────────────────

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "emergency-triage-env",
        "version": "1.0.0",
        "description": "OpenEnv-compatible ER triage environment",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    }
