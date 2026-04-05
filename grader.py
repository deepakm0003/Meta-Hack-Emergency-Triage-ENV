"""
Reward function — all 6 components with task-aware scaling.
Fixes applied:
  1. Penalty uses fractional thresholds (0.60/0.40/0.85) with per-task caps
  2. Belief update gives partial credit per patient (not binary)
  3. Stable compute_reward() compatible with main.py call signature
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from scipy.stats import kendalltau


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truth_ranking(patients: List[Dict[str, Any]]) -> List[str]:
    """Return patient IDs sorted by esi_truth ascending (1=most urgent first)."""
    return [p["id"] for p in sorted(patients, key=lambda p: p["esi_truth"])]


def _rank_from_ordering(ordered_ids: List[str], all_ids: List[str]) -> List[int]:
    pos = {pid: i for i, pid in enumerate(ordered_ids)}
    return [pos.get(pid, len(ordered_ids)) for pid in all_ids]


# ── Component 1: Ranking Score (max 0.40) ────────────────────────────────────

def compute_ranking_score(
    patients: List[Dict[str, Any]],
    agent_ranking: List[str],
    outcome_score: Optional[float] = None,
) -> float:
    if outcome_score is not None:
        base = outcome_score * 0.40
    else:
        all_ids = [p["id"] for p in patients]
        truth_order = _truth_ranking(patients)
        agent_ranks = _rank_from_ordering(agent_ranking, all_ids)
        truth_ranks = _rank_from_ordering(truth_order, all_ids)

        tau_component = 0.0
        if len(set(agent_ranks)) >= 2:
            tau, _ = kendalltau(agent_ranks, truth_ranks)
            tau_component = (tau + 1) / 2   # normalise [-1,1] → [0,1]

        # Weighted position partial credit
        n = len(all_ids)
        pos_score = 0.0
        if n > 0:
            agent_pos = {pid: i for i, pid in enumerate(agent_ranking)}
            truth_pos = {pid: i for i, pid in enumerate(truth_order)}
            pos_score = sum(
                1.0 / (1.0 + abs(agent_pos.get(pid, n) - truth_pos.get(pid, n)))
                for pid in all_ids
            ) / n

        base = (0.5 * tau_component + 0.5 * pos_score) * 0.40

    # Bonus: any ESI-1 in top-2
    esi1_ids = [p["id"] for p in patients if p["esi_truth"] == 1]
    if any(pid in set(agent_ranking[:2]) for pid in esi1_ids):
        base = min(base + 0.10, 0.40)

    return round(max(0.0, base), 4)


# ── Component 2: Belief Update Score (max 0.25) ──────────────────────────────

_LAB_CRITICAL_KEYWORDS = [
    "troponin", "lactate", "stemi", "sepsis", "critical",
]


def compute_belief_update(
    patients: List[Dict[str, Any]],
    ranking_before: Optional[List[str]],
    ranking_after: Optional[List[str]],
    task_id: str,
) -> float:
    if task_id == "static_triage":
        return 0.0
    if not ranking_before or not ranking_after:
        return 0.0
    if ranking_before == ranking_after:
        return 0.0

    # Separate ESI-1 and ESI-2 patients with critical labs — tighter keywords
    esi1_lab, esi2_lab = [], []
    for p in patients:
        labs = p.get("labs") or {}
        if not labs:
            continue
        lab_text = str(labs).lower()
        if any(kw in lab_text for kw in _LAB_CRITICAL_KEYWORDS):
            if p["esi_truth"] == 1:
                esi1_lab.append(p["id"])
            elif p["esi_truth"] == 2:
                esi2_lab.append(p["id"])

    if not esi1_lab and not esi2_lab:
        return 0.0

    def _moved_up(pid: str) -> bool:
        if pid not in ranking_before or pid not in ranking_after:
            return False
        return ranking_after.index(pid) < ranking_before.index(pid)

    # ESI-1 critical lab patients count for 80% of belief score
    esi1_moved = sum(1 for pid in esi1_lab if _moved_up(pid))
    esi2_moved = sum(1 for pid in esi2_lab if _moved_up(pid))

    esi1_ratio = (esi1_moved / len(esi1_lab)) if esi1_lab else 0.0
    esi2_ratio = (esi2_moved / len(esi2_lab)) if esi2_lab else 0.0

    # Weighted: ESI-1 movement worth 80%, ESI-2 worth 20%
    combined = 0.80 * esi1_ratio + 0.20 * esi2_ratio
    return round(combined * 0.25, 4)


# ── Component 3: Reasoning Score (max 0.20) ──────────────────────────────────

_CLINICAL_KEYWORDS = [
    "ESI", "triage", "critical", "priority", "GCS", "SpO2",
    "troponin", "sepsis", "airway", "STEMI", "stroke", "lactate",
    "diaphoretic", "hemodynamic", "resuscitation", "deteriorate",
]


def compute_reasoning_score(reasoning: str) -> float:
    if not reasoning:
        return 0.0
    rl = reasoning.lower()
    hits = [k for k in _CLINICAL_KEYWORDS if k.lower() in rl]
    density_score = min(len(hits) / 6, 1.0)
    length_bonus = min(len(reasoning.split()) / 80, 1.0)
    return round((0.7 * density_score + 0.3 * length_bonus) * 0.20, 4)


# ── Component 4: Capacity Score (max 0.10) ───────────────────────────────────

def compute_capacity_score(
    task_id: str,
    allocations: Optional[Dict[str, str]],
    patients: List[Dict[str, Any]],
    icu_beds_available: int = 3,
    resus_bays_available: int = 2,
) -> float:
    if task_id != "mass_casualty":
        return 0.0
    if allocations is None:
        return 0.0

    icu_count   = sum(1 for v in allocations.values() if v.upper() == "ICU")
    resus_count = sum(1 for v in allocations.values() if v.upper() == "RESUS")

    if icu_count > icu_beds_available or resus_count > resus_bays_available:
        return 0.0

    # ESI-1 must not be in general/waiting
    esi1_ids = {p["id"] for p in patients if p["esi_truth"] == 1}
    waiting_esi1 = sum(
        1 for pid, loc in allocations.items()
        if pid in esi1_ids and loc.upper() not in ("ICU", "RESUS")
    )
    if waiting_esi1 > 0:
        return 0.0

    return 0.10


# ── Component 5: Tiered Penalty — fractional thresholds, per-task cap ────────

_PENALTY_CAPS = {
    "static_triage":    0.25,
    "late_info_triage": 0.25,
    "mass_casualty":    0.20,   # lower cap at 12-patient scale
}


def compute_penalty(
    agent_ranking: List[str],
    patients: List[Dict[str, Any]],
    task_id: str,
) -> float:
    n = len(agent_ranking)
    if n == 0:
        return 0.0

    esi1_ids = [p["id"] for p in patients if p["esi_truth"] == 1]
    esi2_ids = [p["id"] for p in patients if p["esi_truth"] == 2]
    penalty  = 0.0

    for pid in esi1_ids:
        if pid not in agent_ranking:
            continue
        frac = agent_ranking.index(pid) / n
        if frac >= 0.60:      # ESI-1 in bottom 40% — critical miss
            penalty += 0.25
        elif frac >= 0.40:    # ESI-1 in middle 20% — partial miss
            penalty += 0.10

    for pid in esi2_ids:
        if pid not in agent_ranking:
            continue
        frac = agent_ranking.index(pid) / n
        if frac >= 0.85:      # ESI-2 only penalised if buried at very bottom
            penalty += 0.08

    cap = _PENALTY_CAPS.get(task_id, 0.25)
    return round(min(penalty, cap), 4)


# ── Component 6: Consistency Bonus (max 0.05) ────────────────────────────────

def compute_consistency_bonus(ranking_history: List[List[str]]) -> float:
    if len(ranking_history) < 3:
        return 0.0
    last_three = [tuple(r[:3]) for r in ranking_history[-3:]]
    return 0.05 if len(set(last_three)) == 1 else 0.0


# ── Master Grader ─────────────────────────────────────────────────────────────

def compute_reward(
    agent_ranking: List[str],
    reasoning: str,
    patients: List[Dict[str, Any]],
    task_id: str,
    allocations: Optional[Dict[str, str]] = None,
    icu_beds_available: int = 3,
    resus_bays_available: int = 2,
    ranking_before: Optional[List[str]] = None,
    ranking_after: Optional[List[str]] = None,
    outcome_score: Optional[float] = None,
    ranking_history: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    history = ranking_history or []

    # 1. Ranking score
    r_score = compute_ranking_score(patients, agent_ranking, outcome_score)

    # 2. Belief update / stability score
    if task_id == "static_triage":
        # No labs on easy task — use stability bonus: ESI-1 in top-2
        esi1_ids = [p["id"] for p in patients if p["esi_truth"] == 1]
        b_score = 0.25 if any(pid in agent_ranking[:2] for pid in esi1_ids) else 0.0
    else:
        # Use the ranking two steps back vs current for reliable before/after comparison.
        # Falls back to ranking_before passed by main.py if history is shallow.
        if len(history) >= 2:
            before = history[-2]
        else:
            before = ranking_before   # provided by main.py at inject_step boundary
        b_score = compute_belief_update(patients, before, agent_ranking, task_id)

    # 3. Reasoning score
    rs_score = compute_reasoning_score(reasoning)

    # 4. Capacity score
    cap_score = compute_capacity_score(
        task_id, allocations, patients, icu_beds_available, resus_bays_available
    )

    # 5. Consistency bonus
    cons_bonus = compute_consistency_bonus(history)

    # 6. Penalty
    penalty = compute_penalty(agent_ranking, patients, task_id)

    total = r_score + b_score + rs_score + cap_score + cons_bonus - penalty
    total = round(max(0.0, min(1.0, total)), 4)

    return {
        "reward": total,
        "reward_breakdown": {
            "ranking_score":         r_score,
            "belief_update_score":   b_score,
            "reasoning_score":       rs_score,
            "capacity_score":        cap_score,
            "consistency_bonus":     cons_bonus,
            "critical_miss_penalty": penalty,
        },
    }

