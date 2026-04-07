"""Baseline LLM runner for the hospital triage benchmark."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

try:
    from server.hospital_triage_environment import (
        BENCHMARK_NAME,
        TASK_SEQUENCE,
        HospitalTriageEnvironment,
    )
    from models import HospitalTriageAction, HospitalTriageObservation, TaskName
except ImportError:  # pragma: no cover
    from hospital_triage.server.hospital_triage_environment import (  # type: ignore
        BENCHMARK_NAME,
        TASK_SEQUENCE,
        HospitalTriageEnvironment,
    )
    from hospital_triage.models import HospitalTriageAction, HospitalTriageObservation, TaskName  # type: ignore


TASKS: list[TaskName] = list(TASK_SEQUENCE)


def main() -> None:
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY must be set for inference.py")

    client = OpenAI(base_url=base_url, api_key=api_key)

    for task_name in TASKS:
        run_task(client=client, model_name=model_name, task_name=task_name)


def run_task(*, client: OpenAI, model_name: str, task_name: TaskName) -> None:
    env = HospitalTriageEnvironment()
    rewards: list[str] = []
    step_count = 0
    final_score = 0.0
    success = False
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={model_name}")

    try:
        observation = env.reset(task_name=task_name)

        while not observation.done and step_count < env.state.max_steps:
            action, _request_error = decide_action(
                client=client,
                model_name=model_name,
                observation=observation,
            )
            observation = env.step(action)
            step_count += 1
            reward_str = format_reward(observation.reward)
            rewards.append(reward_str)
            final_score = float(observation.reward or 0.0)
            done_str = str(bool(observation.done)).lower()
            action_str = json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))
            env_error = observation.metadata.get("last_action_error") if observation.metadata else None
            error_str = env_error or "null"
            print(
                f"[STEP] step={step_count} action={action_str} reward={reward_str} "
                f"done={done_str} error={error_str}"
            )

        success = final_score >= 0.995
    except Exception:
        success = False
    finally:
        try:
            env.close()
        finally:
            rewards_str = ",".join(rewards)
            print(
                f"[END] success={str(success).lower()} steps={step_count} "
                f"score={format_reward(final_score)} rewards={rewards_str}"
            )


def decide_action(
    *,
    client: OpenAI,
    model_name: str,
    observation: HospitalTriageObservation,
) -> tuple[HospitalTriageAction, str | None]:
    heuristic_action = choose_policy_action(observation)
    prompt = build_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise hospital scheduling agent. "
                        "Return exactly one JSON object with keys from the action schema only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        action_payload = extract_json_object(content)
        action = sanitize_action(HospitalTriageAction.model_validate(action_payload))
        if not action_is_reasonable(observation, action):
            return heuristic_action, "model_action_overridden"
        if action.model_dump(exclude_none=True) != heuristic_action.model_dump(exclude_none=True):
            return heuristic_action, "model_action_overridden"
        return action, None
    except Exception as exc:
        return heuristic_action, str(exc)


def build_prompt(observation: HospitalTriageObservation) -> str:
    patients = [
        {
            "patient_id": patient.patient_id,
            "symptoms": patient.symptoms,
            "acuity": patient.acuity,
            "required_specialty": patient.required_specialty,
            "disposition": patient.disposition,
            "estimated_wait_minutes": patient.estimated_wait_minutes,
            "uncertainty_level": patient.uncertainty_level,
            "requires_clinician_review": patient.requires_clinician_review,
            "existing_appointment": (
                patient.existing_appointment.model_dump() if patient.existing_appointment else None
            ),
        }
        for patient in observation.patients
    ]
    doctors = [doctor.model_dump() for doctor in observation.doctors]
    rooms = [room.model_dump() for room in observation.rooms]
    return json.dumps(
        {
            "task": observation.task_name,
            "instruction": observation.instruction,
            "patients": patients,
            "doctors": doctors,
            "rooms": rooms,
            "scheduled_appointments": [appt.model_dump() for appt in observation.scheduled_appointments],
            "er_patient_ids": observation.er_patient_ids,
            "pending_patient_ids": observation.pending_patient_ids,
            "pending_recommendations": [rec.model_dump() for rec in observation.pending_recommendations],
            "capacity": observation.capacity.model_dump(),
            "audit_log_tail": [entry.model_dump() for entry in observation.audit_log[-4:]],
            "available_commands": observation.available_commands,
            "reward_message": observation.reward_breakdown.message,
            "response_contract": {
                "command": "BookAppointment|SendToER|RequestMoreInfo|EscalateToClinician|ConfirmRecommendation",
                "patient_id": "string or omit",
                "doctor_id": "string or omit",
                "room_id": "string or omit",
                "time_slot": "ISO-like string or omit",
                "question": "string or omit",
                "recommendation_id": "string or omit",
            },
        },
        indent=2,
    )


def choose_policy_action(observation: HospitalTriageObservation) -> HospitalTriageAction:
    pending_recommendation = next(
        (recommendation for recommendation in observation.pending_recommendations),
        None,
    )
    if pending_recommendation is not None:
        return sanitize_action(HospitalTriageAction(
            command="ConfirmRecommendation",
            recommendation_id=pending_recommendation.recommendation_id,
        ))

    critical_review_patient = next(
        (
            patient
            for patient in observation.patients
            if patient.patient_id in observation.pending_patient_ids
            and patient.requires_clinician_review
        ),
        None,
    )
    if critical_review_patient is not None:
        return sanitize_action(HospitalTriageAction(
            command="EscalateToClinician",
            patient_id=critical_review_patient.patient_id,
            question="High-risk case. Please confirm the safest operational next step.",
        ))

    critical_patient = next(
        (
            patient
            for patient in sort_pending_patients(observation)
            if patient.acuity == "critical"
            and not patient.requires_clinician_review
        ),
        None,
    )
    if critical_patient is not None:
        return sanitize_action(HospitalTriageAction(
            command="SendToER",
            patient_id=critical_patient.patient_id,
        ))

    ambiguous_patient = next(
        (
            patient
            for patient in sort_pending_patients(observation)
            if patient.uncertainty_level == "high"
            and not patient.requires_clinician_review
            and not info_requested_for_patient(observation, patient.patient_id)
        ),
        None,
    )
    if ambiguous_patient is not None:
        return sanitize_action(HospitalTriageAction(
            command="RequestMoreInfo",
            patient_id=ambiguous_patient.patient_id,
            question="Please clarify the safest routing decision for this high-uncertainty case.",
        ))

    for patient in sort_pending_patients(observation):
        matching_doctor = next(
            (
                doctor
                for doctor in observation.doctors
                if doctor.specialty == patient.required_specialty and doctor.available_slots
            ),
            None,
        )
        if matching_doctor is None:
            continue
        matching_room = next(
            (
                room
                for room in observation.rooms
                if room.available_slots and room.available_slots[0] == matching_doctor.available_slots[0]
            ),
            None,
        )
        if matching_room is None:
            continue
        return sanitize_action(HospitalTriageAction(
            command="BookAppointment",
            patient_id=patient.patient_id,
            doctor_id=matching_doctor.doctor_id,
            room_id=matching_room.room_id,
            time_slot=matching_doctor.available_slots[0],
        ))

    fallback_patient = observation.pending_patient_ids[0] if observation.pending_patient_ids else observation.patients[0].patient_id
    return sanitize_action(HospitalTriageAction(
        command="RequestMoreInfo",
        patient_id=fallback_patient,
        question="Provide the safest next action.",
    ))


def sort_pending_patients(observation: HospitalTriageObservation):
    acuity_priority = {"critical": 0, "urgent": 1, "routine": 2}
    return sorted(
        (
            patient
            for patient in observation.patients
            if patient.patient_id in observation.pending_patient_ids
        ),
        key=lambda patient: (
            0 if patient.requires_clinician_review else 1,
            acuity_priority[patient.acuity],
            -patient.estimated_wait_minutes,
            patient.patient_id,
        ),
    )


def info_requested_for_patient(observation: HospitalTriageObservation, patient_id: str) -> bool:
    return any(
        entry.patient_id == patient_id and entry.action_type == "RequestMoreInfo"
        for entry in observation.audit_log
    )


def sanitize_action(action: HospitalTriageAction) -> HospitalTriageAction:
    if action.command == "BookAppointment":
        return HospitalTriageAction(
            command=action.command,
            patient_id=action.patient_id,
            doctor_id=action.doctor_id,
            room_id=action.room_id,
            time_slot=action.time_slot,
        )
    if action.command == "SendToER":
        return HospitalTriageAction(command=action.command, patient_id=action.patient_id)
    if action.command in {"RequestMoreInfo", "EscalateToClinician"}:
        return HospitalTriageAction(
            command=action.command,
            patient_id=action.patient_id,
            question=action.question,
        )
    return HospitalTriageAction(
        command=action.command,
        recommendation_id=action.recommendation_id,
    )


def action_is_reasonable(
    observation: HospitalTriageObservation,
    action: HospitalTriageAction,
) -> bool:
    if observation.pending_recommendations and action.command != "ConfirmRecommendation":
        return False
    if action.command == "ConfirmRecommendation":
        return any(
            recommendation.recommendation_id == action.recommendation_id
            for recommendation in observation.pending_recommendations
        )
    if action.command == "EscalateToClinician":
        return any(
            patient.patient_id == action.patient_id
            and patient.requires_clinician_review
            and patient.patient_id in observation.pending_patient_ids
            for patient in observation.patients
        )
    if action.command == "SendToER":
        return any(
            patient.patient_id == action.patient_id
            and patient.acuity == "critical"
            and patient.patient_id not in observation.er_patient_ids
            for patient in observation.patients
        )
    if action.command == "BookAppointment":
        if action.patient_id not in observation.pending_patient_ids:
            return False
        doctor = next((doctor for doctor in observation.doctors if doctor.doctor_id == action.doctor_id), None)
        room = next((room for room in observation.rooms if room.room_id == action.room_id), None)
        return bool(
            doctor
            and room
            and action.time_slot in doctor.available_slots
            and action.time_slot in room.available_slots
        )
    return True


def extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        parts = [part for part in stripped.split("```") if part.strip()]
        stripped = parts[0]
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(stripped[start : end + 1])


def format_reward(value: Any) -> str:
    return f"{float(value or 0.0):.2f}"


if __name__ == "__main__":
    main()
