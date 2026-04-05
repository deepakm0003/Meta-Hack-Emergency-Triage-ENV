"""
60-minute ER outcome simulator for the mass casualty (hard) task.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

DETERIORATION_TIME: Dict[int, int] = {1: 5, 2: 15, 3: 30, 4: 60, 5: 120}
DOCTOR_TIME_PER_PATIENT = 8
SIMULATION_WINDOW = 60


def simulate_outcomes(
    agent_ranking: List[str],
    patients: List[Dict[str, Any]],
    allocations: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    patient_map = {p["id"]: p for p in patients}
    patient_outcomes = []
    clock = 0
    bad_outcomes = 0

    for pid in agent_ranking:
        patient = patient_map.get(pid)
        if patient is None:
            continue

        esi = patient["esi_truth"]
        threshold = DETERIORATION_TIME.get(esi, 60)
        allocation = (allocations or {}).get(pid, "").upper()

        service_time = DOCTOR_TIME_PER_PATIENT
        if allocation in ("ICU", "RESUS") and esi <= 2:
            service_time = max(4, service_time // 2)

        seen_at = clock
        clock += service_time

        deteriorated = clock > SIMULATION_WINDOW or seen_at >= threshold
        if deteriorated:
            bad_outcomes += 1

        patient_outcomes.append({
            "patient_id": pid,
            "esi_truth": esi,
            "seen_at_minute": seen_at,
            "deterioration_threshold": threshold,
            "allocation": allocation or "general",
            "deteriorated": deteriorated,
        })

    total = len(agent_ranking)
    outcome_score = round(1.0 - (bad_outcomes / total), 4) if total > 0 else 0.0

    return {
        "outcome_score": outcome_score,
        "bad_outcomes": bad_outcomes,
        "total_patients": total,
        "patient_outcomes": patient_outcomes,
        "simulation_window_minutes": SIMULATION_WINDOW,
    }
