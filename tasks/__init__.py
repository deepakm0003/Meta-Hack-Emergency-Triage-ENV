from .easy import build_easy_task
from .medium import build_medium_task
from .hard import build_hard_task

TASK_BUILDERS = {
    "static_triage": build_easy_task,
    "late_info_triage": build_medium_task,
    "mass_casualty": build_hard_task,
}

TASK_METADATA = {
    "static_triage": {
        "description": "Rank 5 patients with complete vitals by ESI priority.",
        "difficulty": "easy",
        "max_steps": 5,
        "grader_components": ["ranking_score", "reasoning_score", "critical_miss_penalty"],
        "max_reward": 1.0,
    },
    "late_info_triage": {
        "description": "Rank 6 patients with hidden labs. Lab results injected at step 3.",
        "difficulty": "medium",
        "max_steps": 5,
        "grader_components": ["ranking_score", "belief_update_score", "reasoning_score", "consistency_bonus", "critical_miss_penalty"],
        "max_reward": 1.0,
    },
    "mass_casualty": {
        "description": "12 patients, conflicting notes, ICU/resus constraints, outcome simulation.",
        "difficulty": "hard",
        "max_steps": 5,
        "grader_components": ["ranking_score", "belief_update_score", "reasoning_score", "capacity_score", "consistency_bonus", "critical_miss_penalty"],
        "max_reward": 1.0,
    },
}
