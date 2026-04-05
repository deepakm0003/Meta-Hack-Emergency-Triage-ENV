"""Hard task: mass_casualty — 12 patients, conflicting notes, resource constraints."""
from patient_generator import generate_patients

# 5x ESI-1, 5x ESI-2, 1x ESI-3, 1x ESI-5 for maximum difficulty
HARD_SCENARIO_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14]

RESOURCES = {
    "icu_beds": 3,
    "resus_bays": 2,
    "ventilators": 1,
    "doctors_available": 2,
}


def build_hard_task() -> dict:
    patients = generate_patients(
        HARD_SCENARIO_INDICES,
        hide_labs=True,
        include_conflicting_notes=True,
        shuffle=True,
    )
    return {
        "task_id": "mass_casualty",
        "patients": patients,
        "max_steps": 5,
        "icu_beds_available": RESOURCES["icu_beds"],
        "resus_bays_available": RESOURCES["resus_bays"],
        "inject_labs_at_step": 3,
        "resources": RESOURCES,
    }
