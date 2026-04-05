"""Easy task: static_triage — 5 patients, complete vitals, no info injection."""
from patient_generator import generate_patients

# ESI 1 (STEMI), ESI 2 (Appendicitis), ESI 2 (Asthma), ESI 4 (UTI), ESI 5 (Sprain)
EASY_SCENARIO_INDICES = [0, 5, 6, 12, 14]


def build_easy_task() -> dict:
    patients = generate_patients(
        EASY_SCENARIO_INDICES,
        hide_labs=False,
        include_conflicting_notes=False,
        shuffle=True,
    )
    return {
        "task_id": "static_triage",
        "patients": patients,
        "max_steps": 5,
        "icu_beds_available": None,
        "resus_bays_available": None,
        "inject_labs_at_step": None,
        "resources": None,
    }
