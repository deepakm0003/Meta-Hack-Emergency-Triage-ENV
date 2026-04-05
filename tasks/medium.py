"""Medium task: late_info_triage — 6 patients, labs hidden, injected at step 3."""
from patient_generator import generate_patients

# STEMI(1), Stroke(1), Appendicitis(2), DKA(2), UTI(4), Panic(4)
MEDIUM_SCENARIO_INDICES = [0, 1, 5, 7, 12, 13]


def build_medium_task() -> dict:
    patients = generate_patients(
        MEDIUM_SCENARIO_INDICES,
        hide_labs=True,
        include_conflicting_notes=True,
        shuffle=True,
    )
    return {
        "task_id": "late_info_triage",
        "patients": patients,
        "max_steps": 5,
        "icu_beds_available": None,
        "resus_bays_available": None,
        "inject_labs_at_step": 3,
        "resources": None,
    }
