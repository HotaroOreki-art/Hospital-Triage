---
title: Hospital Triage and Scheduling System
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - healthcare
---

# Hospital Triage and Scheduling System

This OpenEnv benchmark simulates a realistic outpatient triage desk where an agent must make safe scheduling decisions under staffing and room constraints. The environment is fully deterministic, exposes typed Pydantic action and observation models, and scores agent behavior with dense trajectory rewards between `0.0` and `1.0`.

## Real-World Motivation

Hospital front desks and triage coordinators constantly balance urgency, doctor specialty coverage, limited rooms, and schedule disruptions. A useful reinforcement learning benchmark in this domain should reward safe early decisions, expose the operational context an assistant would actually see, and penalize dangerous sequencing errors such as delaying a critical patient with chest pain.

On a personal level, the project was partly inspired by the emotional reality shown in hospital dramas such as HBO's *The Pitt*: patients often experience uncertainty, long waits, and very little visibility into what happens next. This benchmark focuses on whether AI systems can reduce some of that operational stress through safer triage, scheduling, escalation, and communication support.

The real-world feasibility for this kind of system is grounded in existing healthcare research:
- [AHRQ: Machine Learning to Improve Patient Triage in the Emergency Department](https://digital.ahrq.gov/program-overview/research-stories/machine-learning-improve-patient-triage-emergency-department) describes EHR-integrated triage decision support aimed at improving identification of critical illness, admission risk, and fast-track eligibility.
- [Machine learning methods applied to triage in emergency services: A systematic review](https://www.sciencedirect.com/science/article/pii/S1755599X21001476) summarizes evidence that ML can support triage by predicting severity, hospitalization, and critical care needs.
- [Predict, then schedule: Prescriptive analytics approach for machine learning-enabled sequential clinical scheduling](https://www.sciencedirect.com/science/article/pii/S0360835222003357) shows how ML and optimization can be combined to improve clinical appointment scheduling under uncertainty.

This version also models four operational themes that matter in actual care settings:
- Human-in-the-loop confirmation for high-risk recommendations.
- Uncertainty escalation when the agent should defer to a clinician.
- Audit logging so every decision has an explanation trail.
- Wait-time and capacity pressure so the benchmark captures patient experience, not only correctness.

This benchmark now covers five practical workflows:
- Routine appointment booking.
- Multi-patient triage with one critical ER escalation.
- Specialty-safe rescheduling after a doctor calls in sick.
- Ambiguous walk-in clarification before specialty routing.
- Evening-surge coordination across ER, cardiology, neurology, and family medicine.

## Environment Interface

### Action Space

The environment uses a single strict Pydantic action model, `HospitalTriageAction`, with a required `command` field.

Supported commands:
- `BookAppointment`: requires `patient_id`, `doctor_id`, `room_id`, and `time_slot`.
- `SendToER`: requires `patient_id`.
- `RequestMoreInfo`: requires `patient_id` and `question`.
- `EscalateToClinician`: requires `patient_id` and `question`.
- `ConfirmRecommendation`: requires `recommendation_id`.

Example:

```python
HospitalTriageAction(
    command="BookAppointment",
    patient_id="p-routine-1",
    doctor_id="d-family-1",
    room_id="room-exam-1",
    time_slot="2026-04-07T09:00",
)
```

### Observation Space

`HospitalTriageObservation` includes:
- `patients`: symptoms, acuity, required specialty, current disposition, and any canceled appointment.
- `patients[].estimated_wait_minutes`, `uncertainty_level`, and `requires_clinician_review`.
- `doctors`: specialty, availability status, and currently open slots.
- `rooms`: room type and currently open slots.
- `scheduled_appointments`: appointments booked so far.
- `er_patient_ids`: patients already escalated to the ER.
- `pending_patient_ids`: unresolved patients.
- `pending_recommendations`: clinician-review items awaiting confirmation.
- `capacity`: waiting-room count, average wait, ER bed availability, clinic room availability, and pressure level.
- `audit_log`: explanation-rich event trail for agent, clinician, and system actions.
- `reward_breakdown`: structured grading details with component-level signals.
- `instruction`, `task_name`, `reward`, `done`, and metadata.

### Reward Model

Rewards are deterministic and always kept within `[0.0, 1.0]`.

Each task exposes partial credit throughout the trajectory:
- Task 1 rewards correct booking plus wait-time relief and decision logging.
- Task 2 rewards addressing the critical patient first, using clinician confirmation for the risky recommendation, and then scheduling the stable patients.
- Task 3 rewards specialty-correct rescheduling, prioritizing urgent patients earlier, and controlling cumulative wait pressure.
- Task 4 rewards requesting more information before routing an ambiguous urgent walk-in, then clearing the remaining backlog.
- Task 5 rewards handling a septic patient first, preserving ER capacity, and coordinating an evening surge safely.

Dangerous behavior is explicitly penalized:
- In Task 2, any first action that ignores the critical chest-pain patient ends the episode with `0.0`.
- In Task 5, any first action that ignores the septic patient ends the episode with `0.0`.

## Tasks

### Task 1: Easy

Goal: book a routine check-up with an available doctor while clearing waiting-room pressure.

Expected policy behavior:
- Identify the routine patient.
- Choose the available family medicine doctor.
- Use the valid room and time slot.

### Task 2: Medium

Goal: triage three patients, prioritizing a critical chest-pain patient to the ER and scheduling the others.

Expected policy behavior:
- Escalate the critical patient for clinician confirmation first.
- Confirm the recommendation and send the critical patient to the ER.
- Schedule the orthopedic patient with orthopedics.
- Schedule the sore-throat patient with family medicine.

### Task 3: Hard

Goal: reschedule four patients after a doctor calls in sick, matching each patient to the correct specialty without double-booking doctors or rooms while limiting waiting pressure.

Expected policy behavior:
- Read the canceled appointments in the observation.
- Assign each patient to the correct specialist.
- Prioritize the urgent patients earlier.
- Avoid slot collisions across doctors and rooms.

### Task 4: Hard

Goal: clarify an ambiguous abdominal-pain walk-in before routing them, then clear the remaining specialty backlog.

Expected policy behavior:
- Request more information for the high-uncertainty abdominal case first.
- Route the clarified patient to gastroenterology instead of reflexively using the ER.
- Schedule the endocrinology and dermatology patients into their correct same-day slots.

### Task 5: Hard

Goal: manage an evening surge by escalating a septic patient, preserving limited ER capacity, and routing the rest of the queue safely.

Expected policy behavior:
- Escalate the septic patient for clinician confirmation first.
- Confirm the ER recommendation.
- Request more information for the ambiguous arrhythmia walk-in before booking cardiology.
- Schedule the neurology and family medicine patients without leaving backlog unresolved.

## Running Locally

### Install

```bash
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

### Start the OpenEnv Server

```bash
python -m server.app
```

When the web interface is enabled, open `/web` and use the `Task Selector` tab to load any of the five benchmark scenarios before stepping through them in the standard playground.

Two interaction tips for demos:
- Every task starts at `reward = 0.00` on reset because no progress has been made yet.
- After loading a task in `Task Selector`, switch to `Playground` and click `Step` with a valid action. Do not click `Playground Reset` unless you want to return to the default Task 1 scenario.

### Quick Direct Test

```python
from server.hospital_triage_environment import HospitalTriageEnvironment
from models import HospitalTriageAction

env = HospitalTriageEnvironment()
obs = env.reset(task_name="task_2_multi_patient_triage")
obs = env.step(
    HospitalTriageAction(
        command="EscalateToClinician",
        patient_id="p-critical-1",
        question="Please confirm the safest next step for this high-risk case.",
    )
)
obs = env.step(HospitalTriageAction(command="ConfirmRecommendation", recommendation_id="rec-critical-1"))
print(obs.reward, obs.reward_breakdown)
```

## Docker

Build:

```bash
docker build -t hospital-triage-openenv -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 hospital-triage-openenv
```

## Baseline Inference Script

The repository includes `inference.py` in the project root. It uses the standard `openai` Python client and reads these environment variables:
- `API_BASE_URL` with default `https://router.huggingface.co/v1`
- `MODEL_NAME` with default `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN`
- `OPENAI_API_KEY` is also accepted as a local-development fallback, but the official hackathon flow should use `HF_TOKEN`.

Run it like this:

```bash
HF_TOKEN=... python inference.py
```

The script emits logs in this exact format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

## Baseline Scores

Measured with the included `inference.py` baseline runner:
- Task 1: `1.00`
- Task 2: `1.00`
- Task 3: `1.00`
- Task 4: `1.00`
- Task 5: `1.00`

The script runs in well under the 20 minute inference limit on the current 5-task benchmark, and each task has a deterministic maximum score of `1.00` with a partial-credit trajectory along the way.

## Project Structure

```text
hospital_triage/
|- __init__.py
|- client.py
|- inference.py
|- models.py
|- openenv.yaml
|- pyproject.toml
|- README.md
`- server/
   |- app.py
   |- Dockerfile
   |- hospital_triage_environment.py
   `- requirements.txt
```
