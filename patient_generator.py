"""
Synthetic patient scenario builder — 15 clinical archetypes, ESI 1-5.
Includes conflicting_notes for medium/hard tasks to test agent reasoning.
"""
import copy
import random
from typing import List, Dict, Any, Optional

SCENARIO_TEMPLATES: List[Dict[str, Any]] = [
    # ── ESI 1 ─────────────────────────────────────────────────────────────────
    {
        "scenario": "STEMI",
        "age": 67,
        "chief_complaint": "Crushing chest pain radiating to the left arm, onset 45 min ago",
        "vitals": {"bp": "180/110", "hr": 112, "rr": 22, "spo2": 94, "temp": 37.1, "gcs": 15},
        "nurse_note": "Patient diaphoretic, anxious, clutching chest. History of MI 3 years ago.",
        "conflicting_notes": {
            "nurse_A": "Patient appears stable, pain has slightly reduced, vitals holding.",
            "nurse_B": "BP trending up, increasing diaphoresis — looks like active STEMI, needs cath lab NOW."
        },
        "labs_truth": {"troponin": "2.4 ng/mL (CRITICAL HIGH)", "BNP": "650 pg/mL (HIGH)", "ECG": "ST elevation V1-V4"},
        "esi_truth": 1,
    },
    {
        "scenario": "Stroke",
        "age": 72,
        "chief_complaint": "Sudden left-sided facial droop and arm weakness, slurred speech",
        "vitals": {"bp": "195/105", "hr": 88, "rr": 18, "spo2": 96, "temp": 37.0, "gcs": 13},
        "nurse_note": "FAST exam positive. Symptom onset ~90 min ago. Last known well 2 hours ago.",
        "conflicting_notes": {
            "nurse_A": "Patient communicating somewhat, vitals seem ok for age.",
            "nurse_B": "GCS dropping — was 15 at triage, now 13. Facial droop worsening. tPA window closing."
        },
        "labs_truth": {"glucose": "142 mg/dL", "INR": "1.1", "CT_head": "No hemorrhage, thrombus suspected L MCA"},
        "esi_truth": 1,
    },
    {
        "scenario": "Sepsis",
        "age": 58,
        "chief_complaint": "High fever, confusion, and hypotension for 6 hours",
        "vitals": {"bp": "88/52", "hr": 126, "rr": 28, "spo2": 91, "temp": 39.8, "gcs": 12},
        "nurse_note": "Known diabetic. Recent UTI treated with antibiotics one week ago. Looks toxic.",
        "conflicting_notes": {
            "nurse_A": "Patient says they feel a bit better since fluids started, temp down slightly.",
            "nurse_B": "Lactate CRITICAL — 4.2. Still hypotensive after 1L bolus. Septic shock trajectory."
        },
        "labs_truth": {"lactate": "4.2 mmol/L (CRITICAL HIGH)", "WBC": "22,000 (HIGH)", "procalcitonin": "18 ng/mL (CRITICAL)", "cultures": "Pending gram-negative"},
        "esi_truth": 1,
    },
    {
        "scenario": "Ectopic Pregnancy",
        "age": 28,
        "chief_complaint": "Sharp lower abdominal pain, dizziness, missed period",
        "vitals": {"bp": "80/50", "hr": 130, "rr": 24, "spo2": 96, "temp": 36.9, "gcs": 14},
        "nurse_note": "Female. LMP 7 weeks ago. Positive urine hCG. Diaphoretic, pain 9/10. Peritoneal signs.",
        "conflicting_notes": {
            "nurse_A": "Patient initially denied pregnancy but test positive — history unreliable.",
            "nurse_B": "BP crashing — 80/50 and dropping. Probable ruptured ectopic. Needs OR immediately."
        },
        "labs_truth": {"beta_hCG": "4500 IU/L (HIGH)", "US": "Free fluid in pelvis, no IUP — ruptured ectopic", "Hgb": "7.2 g/dL (LOW)"},
        "esi_truth": 1,
    },
    {
        "scenario": "Anaphylaxis",
        "age": 34,
        "chief_complaint": "Bee sting 10 min ago, throat tightening, difficulty breathing",
        "vitals": {"bp": "88/58", "hr": 124, "rr": 28, "spo2": 88, "temp": 37.0, "gcs": 14},
        "nurse_note": "Urticaria over trunk and face. Audible stridor. History of anaphylaxis — no epi-pen available.",
        "conflicting_notes": {
            "nurse_A": "Patient calm, says throat feels 'just a bit tight', may be anxiety.",
            "nurse_B": "Stridor audible 3 feet away. SpO2 88% and falling. Airway emergency — get anesthesia NOW."
        },
        "labs_truth": {"tryptase": "28 ng/mL (CRITICAL HIGH)", "ABG": "pH 7.38, pO2 58 mmHg (LOW)", "treatment": "Epinephrine 0.3mg IM given"},
        "esi_truth": 1,
    },
    # ── ESI 2 ─────────────────────────────────────────────────────────────────
    {
        "scenario": "Appendicitis",
        "age": 24,
        "chief_complaint": "Right lower quadrant pain worsening over 12 hours, nausea",
        "vitals": {"bp": "122/78", "hr": 98, "rr": 18, "spo2": 99, "temp": 38.3, "gcs": 15},
        "nurse_note": "Rebound tenderness present. Pain 8/10. Guarding on palpation. No bowel sounds.",
        "conflicting_notes": {
            "nurse_A": "Patient ambulatory, tolerating sips, pain manageable with positioning.",
            "nurse_B": "Abdomen rigid, temp rising — perforation risk high. Needs surgical consult urgently."
        },
        "labs_truth": {"WBC": "16,800 (HIGH)", "CRP": "45 mg/L (HIGH)", "CT_abdomen": "Periappendiceal fat stranding — acute appendicitis"},
        "esi_truth": 2,
    },
    {
        "scenario": "Asthma Attack",
        "age": 19,
        "chief_complaint": "Acute shortness of breath, wheezing, unable to speak in full sentences",
        "vitals": {"bp": "132/84", "hr": 118, "rr": 30, "spo2": 90, "temp": 37.2, "gcs": 15},
        "nurse_note": "Using accessory muscles. Expiratory wheeze bilateral. Inhaler used x3 at home — no relief.",
        "conflicting_notes": {
            "nurse_A": "Patient says they've had worse episodes, expects this will pass like usual.",
            "nurse_B": "CO2 retaining — pCO2 52. Silent chest developing on right. Near-fatal asthma pattern."
        },
        "labs_truth": {"ABG_pH": "7.32 (LOW)", "pCO2": "52 mmHg (HIGH)", "peak_flow": "35% predicted (CRITICAL LOW)"},
        "esi_truth": 2,
    },
    {
        "scenario": "DKA",
        "age": 31,
        "chief_complaint": "Vomiting, abdominal pain, fruity-smelling breath, extreme thirst",
        "vitals": {"bp": "105/68", "hr": 115, "rr": 26, "spo2": 97, "temp": 37.0, "gcs": 14},
        "nurse_note": "Type 1 diabetic, missed insulin. Kussmaul respirations noted. Drowsy.",
        "conflicting_notes": {
            "nurse_A": "Patient alert enough to answer questions, denies chest pain.",
            "nurse_B": "pH 7.21 and glucose 498 — severe DKA. GCS declining. Needs insulin drip and close monitoring."
        },
        "labs_truth": {"glucose": "498 mg/dL (CRITICAL HIGH)", "pH": "7.21 (LOW)", "ketones": "Large (POSITIVE)", "bicarb": "9 mEq/L (CRITICAL LOW)"},
        "esi_truth": 2,
    },
    {
        "scenario": "Head Trauma",
        "age": 45,
        "chief_complaint": "Head injury after fall from ladder, brief loss of consciousness",
        "vitals": {"bp": "148/92", "hr": 76, "rr": 16, "spo2": 98, "temp": 37.0, "gcs": 14},
        "nurse_note": "Patient confused about event. Pupils equal and reactive. Vomited once. Scalp laceration 4cm.",
        "conflicting_notes": {
            "nurse_A": "Patient oriented to person and place, GCS stable, no focal deficits noted.",
            "nurse_B": "CT shows small epidural hematoma — neurosurgery paged. GCS can drop fast with expanding bleed."
        },
        "labs_truth": {"CT_head": "Small epidural hematoma right temporal (7mm)", "coags": "Normal", "ethanol": "0.12 g/dL (elevated)"},
        "esi_truth": 2,
    },
    {
        "scenario": "Hypertensive Emergency",
        "age": 55,
        "chief_complaint": "Severe headache, blurred vision, BP extremely high",
        "vitals": {"bp": "240/140", "hr": 92, "rr": 18, "spo2": 97, "temp": 37.1, "gcs": 15},
        "nurse_note": "Known hypertensive, ran out of medications 1 week ago. Papilledema on fundoscopy. No focal neuro deficits yet.",
        "conflicting_notes": {
            "nurse_A": "Patient says headache is 'usual' for them when stressed, BP sometimes runs high.",
            "nurse_B": "BP 240/140 — hypertensive emergency. End-organ damage imminent. Papilledema confirmed."
        },
        "labs_truth": {"creatinine": "2.4 mg/dL (HIGH)", "BNP": "420 pg/mL (HIGH)", "urinalysis": "3+ protein, RBC casts"},
        "esi_truth": 2,
    },
    {
        "scenario": "Pulmonary Embolism",
        "age": 48,
        "chief_complaint": "Sudden onset shortness of breath, right-sided chest pain, recent long-haul flight",
        "vitals": {"bp": "108/72", "hr": 118, "rr": 26, "spo2": 91, "temp": 37.3, "gcs": 15},
        "nurse_note": "Returned from 14-hour flight 2 days ago. Right calf swollen and tender. Pleuritic chest pain.",
        "conflicting_notes": {
            "nurse_A": "Patient anxious flier — likely anxiety. SpO2 improved slightly with supplemental O2.",
            "nurse_B": "D-dimer grossly elevated, HR 118, SpO2 91%. Wells score 7 — high probability PE. CT-PA urgently needed."
        },
        "labs_truth": {"D_dimer": "8.4 ug/mL (CRITICAL HIGH)", "troponin": "0.08 ng/mL (mildly elevated)", "CT_PA": "Bilateral pulmonary emboli, right heart strain"},
        "esi_truth": 2,
    },
    # ── ESI 3 ─────────────────────────────────────────────────────────────────
    {
        "scenario": "Acute Psychosis",
        "age": 26,
        "chief_complaint": "Agitated, disorganized speech, paranoid delusions, brought by police",
        "vitals": {"bp": "138/88", "hr": 102, "rr": 18, "spo2": 99, "temp": 37.4, "gcs": 15},
        "nurse_note": "No medical emergency identified. Requires psychiatric evaluation. No known drug use. First episode.",
        "conflicting_notes": {
            "nurse_A": "Medically stable — psych consult appropriate, no urgency.",
            "nurse_B": "Agitation escalating — may become danger to self/others. Needs rapid assessment and possible sedation."
        },
        "labs_truth": {"tox_screen": "Negative", "glucose": "94 mg/dL", "TSH": "Normal", "CT_head": "No acute intracranial pathology"},
        "esi_truth": 3,
    },
    # ── ESI 4 ─────────────────────────────────────────────────────────────────
    {
        "scenario": "UTI",
        "age": 34,
        "chief_complaint": "Burning on urination, frequency, mild lower back pain for 2 days",
        "vitals": {"bp": "118/76", "hr": 80, "rr": 16, "spo2": 99, "temp": 37.8, "gcs": 15},
        "nurse_note": "No systemic symptoms. Alert and oriented. Denies chills or rigors.",
        "conflicting_notes": {
            "nurse_A": "Afebrile at triage — uncomplicated UTI, can wait.",
            "nurse_B": "Temp now 37.8 and trending up — monitor for pyelonephritis."
        },
        "labs_truth": {"UA": "Nitrites positive, leukocyte esterase 3+, WBC 50-100", "culture": "E. coli susceptible to nitrofurantoin"},
        "esi_truth": 4,
    },
    {
        "scenario": "Panic Attack",
        "age": 28,
        "chief_complaint": "Palpitations, shortness of breath, tingling in hands, feeling of doom",
        "vitals": {"bp": "138/88", "hr": 104, "rr": 22, "spo2": 99, "temp": 37.0, "gcs": 15},
        "nurse_note": "Alert, anxious. No cardiac history. Episodes resolve spontaneously. ECG normal sinus tachycardia.",
        "conflicting_notes": {
            "nurse_A": "Classic panic — reassurance and observation should suffice.",
            "nurse_B": "Must rule out PE and ACS first given HR and dyspnea — troponin and D-dimer pending."
        },
        "labs_truth": {"ECG": "Normal sinus tachycardia, no ischemic changes", "troponin": "Negative", "D_dimer": "Normal"},
        "esi_truth": 4,
    },
    # ── ESI 5 ─────────────────────────────────────────────────────────────────
    {
        "scenario": "Sprained Ankle",
        "age": 22,
        "chief_complaint": "Twisted ankle playing basketball, pain and swelling, 2 hours ago",
        "vitals": {"bp": "120/78", "hr": 72, "rr": 14, "spo2": 100, "temp": 36.8, "gcs": 15},
        "nurse_note": "Ambulatory with difficulty. Lateral ankle swelling. Ottawa rules negative for fracture.",
        "conflicting_notes": {
            "nurse_A": "Non-urgent — ice and elevation while waiting.",
            "nurse_B": "Ottawa rules borderline — X-ray to be safe."
        },
        "labs_truth": {"X_ray": "No fracture. Soft tissue swelling Grade II lateral sprain."},
        "esi_truth": 5,
    },
]


def _make_patient_id(index: int) -> str:
    return f"P{index + 1:03d}"


def generate_patients(
    scenario_indices: List[int],
    hide_labs: bool = False,
    include_conflicting_notes: bool = False,
    shuffle: bool = True,
) -> List[Dict[str, Any]]:
    selected = [copy.deepcopy(SCENARIO_TEMPLATES[i]) for i in scenario_indices]
    if shuffle:
        random.shuffle(selected)

    patients = []
    for idx, template in enumerate(selected):
        patient: Dict[str, Any] = {
            "id": _make_patient_id(idx),
            "age": template["age"],
            "chief_complaint": template["chief_complaint"],
            "vitals": copy.deepcopy(template["vitals"]),
            "nurse_note": template["nurse_note"],
            "labs": None if hide_labs else copy.deepcopy(template["labs_truth"]),
            "esi_truth": template["esi_truth"],
            "_scenario": template["scenario"],
            "_labs_truth": copy.deepcopy(template["labs_truth"]),
        }
        if include_conflicting_notes:
            patient["conflicting_notes"] = copy.deepcopy(template.get("conflicting_notes", {}))
        patients.append(patient)
    return patients


def inject_labs(patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for p in patients:
        if p.get("labs") is None:
            p["labs"] = copy.deepcopy(p.get("_labs_truth", {}))
    return patients


def get_public_patient(patient: Dict[str, Any], include_conflicting: bool = False) -> Dict[str, Any]:
    pub = {
        "id": patient["id"],
        "age": patient["age"],
        "chief_complaint": patient["chief_complaint"],
        "vitals": patient["vitals"],
        "nurse_note": patient["nurse_note"],
        "labs": patient["labs"],
    }
    if include_conflicting and "conflicting_notes" in patient:
        pub["conflicting_notes"] = patient["conflicting_notes"]
    return pub
