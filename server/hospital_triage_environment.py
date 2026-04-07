# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic hospital triage and scheduling benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        AppointmentDetails,
        AuditLogEntry,
        CapacitySnapshot,
        DoctorScheduleObservation,
        HospitalTriageAction,
        HospitalTriageObservation,
        HospitalTriageState,
        PatientObservation,
        PendingRecommendation,
        RewardBreakdown,
        RewardComponent,
        RoomAvailabilityObservation,
        TaskName,
    )
except ImportError:  # pragma: no cover
    from models import (
        AppointmentDetails,
        AuditLogEntry,
        CapacitySnapshot,
        DoctorScheduleObservation,
        HospitalTriageAction,
        HospitalTriageObservation,
        HospitalTriageState,
        PatientObservation,
        PendingRecommendation,
        RewardBreakdown,
        RewardComponent,
        RoomAvailabilityObservation,
        TaskName,
    )


BENCHMARK_NAME = "hospital_triage"
TASK_SEQUENCE: tuple[TaskName, ...] = (
    "task_1_routine_checkup",
    "task_2_multi_patient_triage",
    "task_3_specialty_reschedule",
    "task_4_ambiguous_walk_in",
    "task_5_evening_surge_coordination",
)
TASK_METADATA: dict[TaskName, dict[str, str]] = {
    "task_1_routine_checkup": {
        "label": "Task 1: Routine Check-Up",
        "difficulty": "Easy",
    },
    "task_2_multi_patient_triage": {
        "label": "Task 2: Multi-Patient Triage",
        "difficulty": "Medium",
    },
    "task_3_specialty_reschedule": {
        "label": "Task 3: Specialty Reschedule",
        "difficulty": "Hard",
    },
    "task_4_ambiguous_walk_in": {
        "label": "Task 4: Ambiguous Walk-In",
        "difficulty": "Hard",
    },
    "task_5_evening_surge_coordination": {
        "label": "Task 5: Evening Surge Coordination",
        "difficulty": "Hard",
    },
}


@dataclass(frozen=True)
class TaskScenario:
    task_name: TaskName
    instruction: str
    max_steps: int
    er_bed_capacity: int
    patients: list[PatientObservation]
    doctors: list[DoctorScheduleObservation]
    rooms: list[RoomAvailabilityObservation]
    info_bank: dict[str, str]
    recommendation_bank: dict[str, PendingRecommendation]
    grader_name: str


class HospitalTriageEnvironment(Environment):
    """Hospital triage environment with deterministic tasks and scoring."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task_map = self._build_task_map()
        self._scenario = self._task_map["task_1_routine_checkup"]
        self._patients: list[PatientObservation] = []
        self._doctors: list[DoctorScheduleObservation] = []
        self._rooms: list[RoomAvailabilityObservation] = []
        self._state = HospitalTriageState(episode_id=str(uuid4()))
        self.reset(task_name="task_1_routine_checkup")

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: TaskName | None = None,
        **_: object,
    ) -> HospitalTriageObservation:
        """Reset the environment to one of the deterministic benchmark tasks."""
        del seed
        scenario = self._task_map[task_name or "task_1_routine_checkup"]
        self._scenario = scenario
        self._patients = [patient.model_copy(deep=True) for patient in scenario.patients]
        self._doctors = [doctor.model_copy(deep=True) for doctor in scenario.doctors]
        self._rooms = [room.model_copy(deep=True) for room in scenario.rooms]
        self._state = HospitalTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            benchmark=BENCHMARK_NAME,
            task_name=scenario.task_name,
            current_score=0.0,
            max_steps=scenario.max_steps,
            dangerous_outcome=False,
            pending_patient_ids=[patient.patient_id for patient in self._patients],
        )
        self._sync_capacity()
        self._append_audit(
            actor="system",
            action_type="ResetEpisode",
            detail=f"Loaded {scenario.task_name}.",
            explanation="The environment starts from a clean deterministic state.",
            outcome="ready",
        )
        reward = self._grade_current_state()
        self._state.current_score = reward.score
        return self._build_observation(message="Environment reset.", reward=reward, done=False)

    def step(
        self,
        action: HospitalTriageAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> HospitalTriageObservation:
        """Apply a scheduling or triage action and return the next observation."""
        del timeout_s
        self._state.step_count += 1
        self._state.last_action_error = None
        error_message: str | None = None

        try:
            if action.command == "BookAppointment":
                action_message = self._handle_book_appointment(action)
            elif action.command == "SendToER":
                action_message = self._handle_send_to_er(action)
            elif action.command == "RequestMoreInfo":
                action_message = self._handle_request_more_info(action)
            elif action.command == "EscalateToClinician":
                action_message = self._handle_escalate_to_clinician(action)
            else:
                action_message = self._handle_confirm_recommendation(action)
        except ValueError as exc:
            action_message = "Action rejected."
            error_message = str(exc)
            self._state.last_action_error = error_message
            self._append_audit(
                actor="system",
                action_type="RejectedAction",
                detail=action.command,
                explanation=error_message,
                outcome="rejected",
                patient_id=action.patient_id,
            )

        self._advance_wait_clock()
        self._state.pending_patient_ids = self._compute_pending_patient_ids()
        self._sync_capacity()

        action_summary = self._summarize_action(action, error_message)
        self._state.action_history.append(action_summary)
        self._state.last_action_summary = action_summary

        reward = self._grade_current_state()
        self._state.current_score = reward.score
        self._state.dangerous_outcome = reward.dangerous
        done = (
            reward.dangerous
            or reward.score >= 0.995
            or self._state.step_count >= self._state.max_steps
        )
        message = action_message if error_message is None else f"{action_message} {error_message}"
        return self._build_observation(message=message, reward=reward, done=done)

    @property
    def state(self) -> HospitalTriageState:
        """Expose deterministic state for debugging and evaluation."""
        return self._state

    def available_tasks(self) -> list[TaskName]:
        """Return the supported benchmark tasks in a stable order."""
        return list(TASK_SEQUENCE)

    def task_catalog(self) -> list[dict[str, str | int]]:
        """Return task metadata for the web selector and documentation."""
        catalog: list[dict[str, str | int]] = []
        for task_name in TASK_SEQUENCE:
            scenario = self._task_map[task_name]
            metadata = TASK_METADATA[task_name]
            catalog.append(
                {
                    "task_name": task_name,
                    "label": metadata["label"],
                    "difficulty": metadata["difficulty"],
                    "instruction": scenario.instruction,
                    "max_steps": scenario.max_steps,
                }
            )
        return catalog

    def _grade_current_state(self) -> RewardBreakdown:
        return getattr(self, self._scenario.grader_name)()

    def _build_observation(
        self,
        *,
        message: str,
        reward: RewardBreakdown,
        done: bool,
    ) -> HospitalTriageObservation:
        for patient in self._patients:
            patient.disposition = self._patient_disposition(patient.patient_id)

        self._sync_capacity()
        return HospitalTriageObservation(
            benchmark=BENCHMARK_NAME,
            task_name=self._scenario.task_name,
            instruction=self._scenario.instruction,
            message=message,
            patients=[patient.model_copy(deep=True) for patient in self._patients],
            doctors=self._visible_doctors(),
            rooms=self._visible_rooms(),
            scheduled_appointments=[appt.model_copy(deep=True) for appt in self._state.scheduled_appointments],
            er_patient_ids=list(self._state.er_patient_ids),
            pending_patient_ids=self._compute_pending_patient_ids(),
            pending_recommendations=[
                recommendation.model_copy(deep=True)
                for recommendation in self._state.pending_recommendations
            ],
            capacity=self._state.capacity.model_copy(deep=True),
            audit_log=[entry.model_copy(deep=True) for entry in self._state.audit_log],
            reward_breakdown=reward,
            reward=reward.score,
            done=done,
            metadata={
                "step_count": self._state.step_count,
                "dangerous_outcome": reward.dangerous,
                "last_info_response": self._state.last_info_response,
                "last_action_error": self._state.last_action_error,
                "total_wait_minutes_elapsed": self._state.total_wait_minutes_elapsed,
            },
        )

    def _handle_book_appointment(self, action: HospitalTriageAction) -> str:
        appointment_message = self._apply_book_appointment(
            patient_id=action.patient_id,
            doctor_id=action.doctor_id,
            room_id=action.room_id,
            time_slot=action.time_slot,
        )
        patient = self._get_patient(action.patient_id)
        self._append_audit(
            actor="agent",
            action_type="BookAppointment",
            detail=appointment_message,
            explanation=(
                f"Booked the lowest-friction specialty match for {patient.name} "
                "while reducing waiting-room pressure."
            ),
            outcome="scheduled",
            patient_id=patient.patient_id,
        )
        self._state.last_info_response = ""
        return appointment_message

    def _handle_send_to_er(self, action: HospitalTriageAction) -> str:
        patient = self._get_patient(action.patient_id)
        self._apply_send_to_er(patient.patient_id)
        explanation = "Escalated immediately based on critical symptoms."
        outcome = "sent_to_er"
        if patient.requires_clinician_review and not self._has_confirmed_recommendation(patient.patient_id):
            explanation = "Direct ER transfer happened before explicit clinician confirmation."
            outcome = "sent_to_er_without_confirmation"
        self._append_audit(
            actor="agent",
            action_type="SendToER",
            detail=f"Sent {patient.name} to the ER.",
            explanation=explanation,
            outcome=outcome,
            patient_id=patient.patient_id,
        )
        self._state.last_info_response = ""
        return f"Sent {patient.name} to the ER."

    def _handle_request_more_info(self, action: HospitalTriageAction) -> str:
        patient = self._get_patient(action.patient_id)
        response = self._scenario.info_bank.get(
            patient.patient_id,
            f"{patient.name} requires {patient.required_specialty}.",
        )
        self._state.last_info_response = response
        self._append_audit(
            actor="agent",
            action_type="RequestMoreInfo",
            detail=f"Requested more information for {patient.name}.",
            explanation=action.question or "Clarified the next safest action.",
            outcome="information_returned",
            patient_id=patient.patient_id,
        )
        return f"Additional information for {patient.name}: {response}"

    def _handle_escalate_to_clinician(self, action: HospitalTriageAction) -> str:
        patient = self._get_patient(action.patient_id)
        recommendation = self._upsert_recommendation(patient.patient_id)
        if patient.patient_id not in self._state.clinician_reviewed_patient_ids:
            self._state.clinician_reviewed_patient_ids.append(patient.patient_id)

        self._append_audit(
            actor="agent",
            action_type="EscalateToClinician",
            detail=f"Escalated {patient.name} for clinician review.",
            explanation=action.question or "The agent detected uncertainty or elevated risk.",
            outcome="review_requested",
            patient_id=patient.patient_id,
        )
        self._append_audit(
            actor="clinician",
            action_type="IssueRecommendation",
            detail=recommendation.explanation,
            explanation="A clinician confirmed the safest operational next step.",
            outcome=recommendation.recommendation_id,
            patient_id=patient.patient_id,
        )
        self._state.last_info_response = recommendation.explanation
        return (
            f"Clinician review completed for {patient.name}. "
            f"Pending recommendation: {recommendation.recommendation_id}."
        )

    def _handle_confirm_recommendation(self, action: HospitalTriageAction) -> str:
        recommendation = self._get_pending_recommendation(action.recommendation_id)
        patient = self._get_patient(recommendation.patient_id)
        self._state.pending_recommendations = [
            item
            for item in self._state.pending_recommendations
            if item.recommendation_id != recommendation.recommendation_id
        ]
        self._state.confirmed_recommendation_ids.append(recommendation.recommendation_id)
        if recommendation.patient_id not in self._state.clinician_reviewed_patient_ids:
            self._state.clinician_reviewed_patient_ids.append(recommendation.patient_id)

        if recommendation.command == "SendToER":
            outcome_message = self._apply_send_to_er(recommendation.patient_id)
        else:
            outcome_message = self._apply_book_appointment(
                patient_id=recommendation.patient_id,
                doctor_id=recommendation.doctor_id,
                room_id=recommendation.room_id,
                time_slot=recommendation.time_slot,
            )

        self._append_audit(
            actor="clinician",
            action_type="ConfirmRecommendation",
            detail=f"Confirmed {recommendation.recommendation_id} for {patient.name}.",
            explanation=recommendation.explanation,
            outcome="confirmed_and_applied",
            patient_id=patient.patient_id,
        )
        self._state.last_info_response = recommendation.explanation
        return outcome_message

    def _apply_book_appointment(
        self,
        *,
        patient_id: str | None,
        doctor_id: str | None,
        room_id: str | None,
        time_slot: str | None,
    ) -> str:
        patient = self._get_patient(patient_id)
        doctor = self._get_doctor(doctor_id)
        room = self._get_room(room_id)
        if time_slot is None:
            raise ValueError("time_slot is required.")
        if doctor.status != "available":
            raise ValueError(f"Doctor {doctor.doctor_id} is not available.")
        if time_slot not in doctor.available_slots:
            raise ValueError(f"Doctor {doctor.doctor_id} is unavailable at {time_slot}.")
        if time_slot not in room.available_slots:
            raise ValueError(f"Room {room.room_id} is unavailable at {time_slot}.")
        if self._find_appointment_by_doctor_and_slot(doctor.doctor_id, time_slot):
            raise ValueError(f"Doctor {doctor.doctor_id} is already booked at {time_slot}.")
        if self._find_appointment_by_room_and_slot(room.room_id, time_slot):
            raise ValueError(f"Room {room.room_id} is already booked at {time_slot}.")

        self._state.scheduled_appointments = [
            appt for appt in self._state.scheduled_appointments if appt.patient_id != patient.patient_id
        ]
        self._state.scheduled_appointments.append(
            AppointmentDetails(
                patient_id=patient.patient_id,
                doctor_id=doctor.doctor_id,
                doctor_name=doctor.name,
                specialty=doctor.specialty,
                room_id=room.room_id,
                time_slot=time_slot,
                status="scheduled",
            )
        )
        patient.estimated_wait_minutes = 0
        self._state.resolution_steps.setdefault(patient.patient_id, self._state.step_count)
        return f"Scheduled {patient.name} with {doctor.name} in {room.room_id} at {time_slot}."

    def _apply_send_to_er(self, patient_id: str) -> str:
        patient = self._get_patient(patient_id)
        if self._available_er_beds() <= 0 and patient.patient_id not in self._state.er_patient_ids:
            raise ValueError("No ER beds available.")
        if patient.patient_id not in self._state.er_patient_ids:
            self._state.er_patient_ids.append(patient.patient_id)
        self._state.scheduled_appointments = [
            appt for appt in self._state.scheduled_appointments if appt.patient_id != patient.patient_id
        ]
        patient.estimated_wait_minutes = 0
        self._state.resolution_steps.setdefault(patient.patient_id, self._state.step_count)
        return f"Sent {patient.name} to the ER."

    def _grade_task_1(self) -> RewardBreakdown:
        appointment = self._appointment_for_patient("p-routine-1")
        components: list[RewardComponent] = []
        score = 0.0

        if appointment is not None:
            score += 0.20
            components.append(
                RewardComponent(
                    name="patient_addressed",
                    score=0.20,
                    detail="The routine patient has an appointment on the board.",
                )
            )
            if appointment.doctor_id == "d-family-1":
                score += 0.25
                components.append(
                    RewardComponent(
                        name="correct_doctor",
                        score=0.25,
                        detail="The available family medicine doctor was selected.",
                    )
                )
            if appointment.time_slot == "2026-04-07T09:00" and appointment.room_id == "room-exam-1":
                score += 0.25
                components.append(
                    RewardComponent(
                        name="valid_slot",
                        score=0.25,
                        detail="The earliest low-friction slot was used.",
                    )
                )
            if self._patient_wait("p-routine-1") == 0:
                score += 0.15
                components.append(
                    RewardComponent(
                        name="wait_time_relief",
                        score=0.15,
                        detail="The patient no longer contributes to waiting-room stress.",
                    )
                )
            if any(
                entry.action_type == "BookAppointment" and entry.patient_id == "p-routine-1"
                for entry in self._state.audit_log
            ):
                score += 0.15
                components.append(
                    RewardComponent(
                        name="audit_logged",
                        score=0.15,
                        detail="The appointment decision was captured in the audit trail.",
                    )
                )

        message = "Book the routine check-up with the available family medicine doctor."
        if score >= 0.995:
            message = "Task complete."
        return RewardBreakdown(score=min(score, 1.0), dangerous=False, message=message, components=components)

    def _grade_task_2(self) -> RewardBreakdown:
        if self._state.step_count > 0 and not self._state.action_history[0].startswith(("EscalateToClinician:p-critical-1", "SendToER:p-critical-1")):
            return RewardBreakdown(
                score=0.0,
                dangerous=True,
                message="Dangerous delay: the critical chest-pain patient was not addressed first.",
                components=[
                    RewardComponent(
                        name="critical_delay",
                        score=0.0,
                        detail="Critical chest pain must be the first priority.",
                    )
                ],
            )

        score = 0.0
        components: list[RewardComponent] = []
        if self._state.action_history and self._state.action_history[0].startswith("EscalateToClinician:p-critical-1"):
            score += 0.20
            components.append(
                RewardComponent(
                    name="safe_escalation_first",
                    score=0.20,
                    detail="The agent requested clinician review before acting on an uncertain critical case.",
                )
            )
        if "rec-critical-1" in self._state.confirmed_recommendation_ids:
            score += 0.20
            components.append(
                RewardComponent(
                    name="clinician_confirmation",
                    score=0.20,
                    detail="The high-risk recommendation was explicitly confirmed.",
                )
            )
        if "p-critical-1" in self._state.er_patient_ids:
            score += 0.20
            components.append(
                RewardComponent(
                    name="critical_patient_to_er",
                    score=0.20,
                    detail="The chest-pain patient reached the ER.",
                )
            )
        ankle_appt = self._appointment_for_patient("p-ankle-1")
        if ankle_appt and ankle_appt.doctor_id == "d-ortho-1" and self._appointment_is_conflict_free(ankle_appt):
            score += 0.20
            components.append(
                RewardComponent(
                    name="orthopedic_patient_scheduled",
                    score=0.20,
                    detail="The sprained ankle patient was booked with orthopedics.",
                )
            )
        throat_appt = self._appointment_for_patient("p-throat-1")
        if throat_appt and throat_appt.doctor_id == "d-family-2" and self._appointment_is_conflict_free(throat_appt):
            score += 0.20
            components.append(
                RewardComponent(
                    name="routine_patient_scheduled",
                    score=0.20,
                    detail="The sore throat patient was booked with family medicine.",
                )
            )

        message = (
            "Address the critical patient first, confirm the risky recommendation, "
            "then schedule the two stable patients."
        )
        if score >= 0.995:
            message = "Task complete."
        return RewardBreakdown(score=min(score, 1.0), dangerous=False, message=message, components=components)

    def _grade_task_3(self) -> RewardBreakdown:
        expected = {
            "p-cardio-1": ("cardiology", 0.20),
            "p-endo-1": ("endocrinology", 0.20),
            "p-derm-1": ("dermatology", 0.20),
            "p-pulm-1": ("pulmonology", 0.20),
        }
        score = 0.0
        components: list[RewardComponent] = []
        for patient_id, (specialty, weight) in expected.items():
            appointment = self._appointment_for_patient(patient_id)
            if appointment and appointment.specialty == specialty and self._appointment_is_conflict_free(appointment):
                score += weight
                components.append(
                    RewardComponent(
                        name=f"{patient_id}_rescheduled",
                        score=weight,
                        detail=f"{patient_id} was rescheduled with the correct {specialty} clinician.",
                    )
                )

        cardio_step = self._state.resolution_steps.get("p-cardio-1", 999)
        derm_step = self._state.resolution_steps.get("p-derm-1", 999)
        if cardio_step <= 2 and derm_step <= 3:
            score += 0.10
            components.append(
                RewardComponent(
                    name="urgent_patients_prioritized",
                    score=0.10,
                    detail="The urgent reschedules were handled early in the backlog.",
                )
            )
        if self._state.total_wait_minutes_elapsed <= 220 and not self._state.pending_patient_ids:
            score += 0.10
            components.append(
                RewardComponent(
                    name="wait_pressure_controlled",
                    score=0.10,
                    detail="The agent kept cumulative waiting pressure under control.",
                )
            )

        message = "Reschedule all four patients while controlling backlog and avoiding double-booking."
        if score >= 0.995:
            message = "Task complete."
        return RewardBreakdown(score=min(score, 1.0), dangerous=False, message=message, components=components)

    def _grade_task_4(self) -> RewardBreakdown:
        score = 0.0
        components: list[RewardComponent] = []

        info_requested = self._has_requested_info("p-abd-1")
        first_action = self._state.action_history[0] if self._state.action_history else ""
        if first_action.startswith("RequestMoreInfo:p-abd-1"):
            score += 0.20
            components.append(
                RewardComponent(
                    name="ambiguity_clarified_first",
                    score=0.20,
                    detail="The abdominal walk-in was clarified before routing the backlog.",
                )
            )
        elif info_requested:
            score += 0.10
            components.append(
                RewardComponent(
                    name="ambiguity_clarified_late",
                    score=0.10,
                    detail="The agent eventually requested more detail for the ambiguous abdominal case.",
                )
            )

        abdominal_appt = self._appointment_for_patient("p-abd-1")
        if abdominal_appt and abdominal_appt.doctor_id == "d-gi-1" and self._appointment_is_conflict_free(abdominal_appt):
            score += 0.30
            components.append(
                RewardComponent(
                    name="abdominal_case_booked",
                    score=0.30,
                    detail="The abdominal walk-in was booked into the gastroenterology overflow slot.",
                )
            )

        diabetes_appt = self._appointment_for_patient("p-diabetes-1")
        if diabetes_appt and diabetes_appt.doctor_id == "d-endo-3" and self._appointment_is_conflict_free(diabetes_appt):
            score += 0.20
            components.append(
                RewardComponent(
                    name="diabetes_follow_up_booked",
                    score=0.20,
                    detail="The insulin review patient was moved into endocrinology.",
                )
            )

        eczema_appt = self._appointment_for_patient("p-eczema-1")
        if eczema_appt and eczema_appt.doctor_id == "d-derm-3" and self._appointment_is_conflict_free(eczema_appt):
            score += 0.20
            components.append(
                RewardComponent(
                    name="dermatology_patient_booked",
                    score=0.20,
                    detail="The dermatology walk-in was routed to the correct same-day slot.",
                )
            )

        if self._state.total_wait_minutes_elapsed <= 210 and not self._state.pending_patient_ids:
            score += 0.10
            components.append(
                RewardComponent(
                    name="backlog_pressure_controlled",
                    score=0.10,
                    detail="The agent cleared the queue without letting wait pressure spike.",
                )
            )

        message = (
            "Clarify the ambiguous abdominal walk-in before booking them, then clear the "
            "remaining specialty backlog."
        )
        if score >= 0.995:
            message = "Task complete."
        return RewardBreakdown(score=min(score, 1.0), dangerous=False, message=message, components=components)

    def _grade_task_5(self) -> RewardBreakdown:
        if self._state.step_count > 0 and not self._state.action_history[0].startswith(("EscalateToClinician:p-sepsis-1", "SendToER:p-sepsis-1")):
            return RewardBreakdown(
                score=0.0,
                dangerous=True,
                message="Dangerous delay: the septic patient was not addressed before the rest of the surge.",
                components=[
                    RewardComponent(
                        name="sepsis_delay",
                        score=0.0,
                        detail="Possible sepsis must be the first operational priority.",
                    )
                ],
            )

        score = 0.0
        components: list[RewardComponent] = []
        if self._state.action_history and self._state.action_history[0].startswith("EscalateToClinician:p-sepsis-1"):
            score += 0.15
            components.append(
                RewardComponent(
                    name="safe_sepsis_escalation",
                    score=0.15,
                    detail="The agent escalated the septic patient for clinician confirmation first.",
                )
            )

        if "rec-sepsis-1" in self._state.confirmed_recommendation_ids:
            score += 0.15
            components.append(
                RewardComponent(
                    name="sepsis_recommendation_confirmed",
                    score=0.15,
                    detail="The ER recommendation for possible sepsis was explicitly confirmed.",
                )
            )

        if "p-sepsis-1" in self._state.er_patient_ids:
            score += 0.20
            components.append(
                RewardComponent(
                    name="sepsis_patient_to_er",
                    score=0.20,
                    detail="The septic patient reached the ER despite the evening surge.",
                )
            )

        if self._has_requested_info("p-arrhythmia-1"):
            score += 0.15
            components.append(
                RewardComponent(
                    name="arrhythmia_info_requested",
                    score=0.15,
                    detail="The agent clarified the ambiguous arrhythmia case before booking it.",
                )
            )

        arrhythmia_appt = self._appointment_for_patient("p-arrhythmia-1")
        if arrhythmia_appt and arrhythmia_appt.doctor_id == "d-cardio-3" and self._appointment_is_conflict_free(arrhythmia_appt):
            score += 0.15
            components.append(
                RewardComponent(
                    name="arrhythmia_patient_booked",
                    score=0.15,
                    detail="The stable arrhythmia case was routed to cardiology instead of consuming the ER.",
                )
            )

        headache_appt = self._appointment_for_patient("p-headache-1")
        if headache_appt and headache_appt.doctor_id == "d-neuro-1" and self._appointment_is_conflict_free(headache_appt):
            score += 0.10
            components.append(
                RewardComponent(
                    name="neurology_patient_booked",
                    score=0.10,
                    detail="The headache patient was assigned to neurology.",
                )
            )

        followup_appt = self._appointment_for_patient("p-followup-1")
        if followup_appt and followup_appt.doctor_id == "d-family-3" and self._appointment_is_conflict_free(followup_appt):
            score += 0.05
            components.append(
                RewardComponent(
                    name="family_follow_up_booked",
                    score=0.05,
                    detail="The routine follow-up was scheduled into the remaining family medicine slot.",
                )
            )

        if self._state.total_wait_minutes_elapsed <= 340 and not self._state.pending_patient_ids:
            score += 0.05
            components.append(
                RewardComponent(
                    name="surge_pressure_controlled",
                    score=0.05,
                    detail="The evening surge was cleared without excessive cumulative waiting pressure.",
                )
            )

        message = (
            "Address the septic patient first, clarify the arrhythmia walk-in, and then "
            "clear the remaining evening surge backlog."
        )
        if score >= 0.995:
            message = "Task complete."
        return RewardBreakdown(score=min(score, 1.0), dangerous=False, message=message, components=components)

    def _append_audit(
        self,
        *,
        actor: str,
        action_type: str,
        detail: str,
        explanation: str,
        outcome: str,
        patient_id: str | None = None,
    ) -> None:
        self._state.audit_log.append(
            AuditLogEntry(
                event_id=f"evt-{len(self._state.audit_log) + 1}",
                step_count=self._state.step_count,
                actor=actor,
                action_type=action_type,
                patient_id=patient_id,
                detail=detail,
                explanation=explanation,
                outcome=outcome,
                average_wait_minutes=self._compute_average_wait_minutes(),
            )
        )

    def _advance_wait_clock(self) -> None:
        pressure_bonus = {"low": 0, "medium": 5, "high": 10}[self._compute_capacity_snapshot().pressure_level]
        for patient in self._patients:
            if self._is_patient_resolved(patient.patient_id):
                patient.estimated_wait_minutes = 0
                continue
            increment = 10 + pressure_bonus
            if patient.acuity == "urgent":
                increment += 5
            elif patient.acuity == "critical":
                increment += 10
            if patient.uncertainty_level == "high":
                increment += 5
            patient.estimated_wait_minutes += increment
            self._state.total_wait_minutes_elapsed += increment

    def _sync_capacity(self) -> None:
        self._state.pending_patient_ids = self._compute_pending_patient_ids()
        self._state.capacity = self._compute_capacity_snapshot()

    def _compute_capacity_snapshot(self) -> CapacitySnapshot:
        waiting_patients = [
            patient for patient in self._patients if not self._is_patient_resolved(patient.patient_id)
        ]
        waiting_count = len(waiting_patients)
        average_wait = self._compute_average_wait_minutes()
        clinic_rooms_available = sum(1 for room in self._visible_rooms() if room.available_slots)
        pressure_level = "low"
        if waiting_count >= 3 or average_wait >= 80 or clinic_rooms_available == 0:
            pressure_level = "high"
        elif waiting_count >= 2 or average_wait >= 40:
            pressure_level = "medium"
        return CapacitySnapshot(
            waiting_room_patients=waiting_count,
            average_wait_minutes=average_wait,
            er_beds_available=self._available_er_beds(),
            clinic_rooms_available=clinic_rooms_available,
            pressure_level=pressure_level,
        )

    def _compute_average_wait_minutes(self) -> int:
        waiting_values = [
            patient.estimated_wait_minutes
            for patient in self._patients
            if not self._is_patient_resolved(patient.patient_id)
        ]
        if not waiting_values:
            return 0
        return round(sum(waiting_values) / len(waiting_values))

    def _available_er_beds(self) -> int:
        return max(0, self._scenario.er_bed_capacity - len(self._state.er_patient_ids))

    def _appointment_for_patient(self, patient_id: str) -> AppointmentDetails | None:
        for appointment in self._state.scheduled_appointments:
            if appointment.patient_id == patient_id:
                return appointment
        return None

    def _appointment_is_conflict_free(self, appointment: AppointmentDetails) -> bool:
        doctor = self._get_doctor(appointment.doctor_id)
        room = self._get_room(appointment.room_id)
        if appointment.time_slot not in doctor.available_slots or appointment.time_slot not in room.available_slots:
            return False
        for other in self._state.scheduled_appointments:
            if other.patient_id == appointment.patient_id:
                continue
            if other.doctor_id == appointment.doctor_id and other.time_slot == appointment.time_slot:
                return False
            if other.room_id == appointment.room_id and other.time_slot == appointment.time_slot:
                return False
        return True

    def _patient_wait(self, patient_id: str) -> int:
        return self._get_patient(patient_id).estimated_wait_minutes

    def _patient_disposition(self, patient_id: str) -> str:
        if patient_id in self._state.er_patient_ids:
            return "er"
        if self._appointment_for_patient(patient_id) is not None:
            return "scheduled"
        patient = self._get_patient(patient_id)
        if patient.existing_appointment and patient.existing_appointment.status == "canceled":
            return "canceled"
        return "waiting"

    def _is_patient_resolved(self, patient_id: str) -> bool:
        return patient_id in self._state.er_patient_ids or self._appointment_for_patient(patient_id) is not None

    def _compute_pending_patient_ids(self) -> list[str]:
        pending: list[str] = []
        for patient in self._patients:
            if not self._is_patient_resolved(patient.patient_id):
                pending.append(patient.patient_id)
        return pending

    def _visible_doctors(self) -> list[DoctorScheduleObservation]:
        doctors = [doctor.model_copy(deep=True) for doctor in self._doctors]
        for doctor in doctors:
            used_slots = {
                appt.time_slot
                for appt in self._state.scheduled_appointments
                if appt.doctor_id == doctor.doctor_id
            }
            doctor.available_slots = [slot for slot in doctor.available_slots if slot not in used_slots]
        return doctors

    def _visible_rooms(self) -> list[RoomAvailabilityObservation]:
        rooms = [room.model_copy(deep=True) for room in self._rooms]
        for room in rooms:
            used_slots = {
                appt.time_slot
                for appt in self._state.scheduled_appointments
                if appt.room_id == room.room_id
            }
            room.available_slots = [slot for slot in room.available_slots if slot not in used_slots]
        return rooms

    def _find_appointment_by_doctor_and_slot(self, doctor_id: str, time_slot: str) -> AppointmentDetails | None:
        for appointment in self._state.scheduled_appointments:
            if appointment.doctor_id == doctor_id and appointment.time_slot == time_slot:
                return appointment
        return None

    def _find_appointment_by_room_and_slot(self, room_id: str, time_slot: str) -> AppointmentDetails | None:
        for appointment in self._state.scheduled_appointments:
            if appointment.room_id == room_id and appointment.time_slot == time_slot:
                return appointment
        return None

    def _upsert_recommendation(self, patient_id: str) -> PendingRecommendation:
        for recommendation in self._state.pending_recommendations:
            if recommendation.patient_id == patient_id:
                return recommendation
        template = self._scenario.recommendation_bank.get(patient_id)
        if template is None:
            raise ValueError(f"No clinician recommendation available for patient {patient_id}.")
        recommendation = template.model_copy(deep=True)
        self._state.pending_recommendations.append(recommendation)
        return recommendation

    def _get_pending_recommendation(self, recommendation_id: str | None) -> PendingRecommendation:
        if recommendation_id is None:
            raise ValueError("recommendation_id is required.")
        for recommendation in self._state.pending_recommendations:
            if recommendation.recommendation_id == recommendation_id:
                return recommendation
        raise ValueError(f"Unknown recommendation_id: {recommendation_id}")

    def _has_confirmed_recommendation(self, patient_id: str) -> bool:
        recommendation_ids = [
            recommendation.recommendation_id
            for recommendation in self._scenario.recommendation_bank.values()
            if recommendation.patient_id == patient_id
        ]
        return any(recommendation_id in self._state.confirmed_recommendation_ids for recommendation_id in recommendation_ids)

    def _has_requested_info(self, patient_id: str) -> bool:
        return any(
            entry.action_type == "RequestMoreInfo" and entry.patient_id == patient_id
            for entry in self._state.audit_log
        )

    def _get_patient(self, patient_id: str | None) -> PatientObservation:
        if patient_id is None:
            raise ValueError("patient_id is required.")
        for patient in self._patients:
            if patient.patient_id == patient_id:
                return patient
        raise ValueError(f"Unknown patient_id: {patient_id}")

    def _get_doctor(self, doctor_id: str | None) -> DoctorScheduleObservation:
        if doctor_id is None:
            raise ValueError("doctor_id is required.")
        for doctor in self._doctors:
            if doctor.doctor_id == doctor_id:
                return doctor
        raise ValueError(f"Unknown doctor_id: {doctor_id}")

    def _get_room(self, room_id: str | None) -> RoomAvailabilityObservation:
        if room_id is None:
            raise ValueError("room_id is required.")
        for room in self._rooms:
            if room.room_id == room_id:
                return room
        raise ValueError(f"Unknown room_id: {room_id}")

    def _summarize_action(self, action: HospitalTriageAction, error_message: str | None) -> str:
        summary = ":".join(
            [
                action.command,
                action.patient_id or "-",
                action.doctor_id or "-",
                action.room_id or "-",
                action.time_slot or "-",
                action.recommendation_id or "-",
            ]
        )
        return f"{summary}:error" if error_message else summary

    @classmethod
    def _build_task_map(cls) -> dict[TaskName, TaskScenario]:
        return {
            "task_1_routine_checkup": TaskScenario(
                task_name="task_1_routine_checkup",
                instruction=(
                    "Task 1 (Easy): book Priya Shah for a routine family medicine check-up "
                    "while reducing waiting-room pressure."
                ),
                max_steps=3,
                er_bed_capacity=1,
                patients=[
                    PatientObservation(
                        patient_id="p-routine-1",
                        name="Priya Shah",
                        age=34,
                        symptoms=["mild fatigue", "annual wellness follow-up"],
                        acuity="routine",
                        required_specialty="family_medicine",
                        notes=["No red flags. Prefers earliest available slot."],
                        estimated_wait_minutes=35,
                        uncertainty_level="low",
                    )
                ],
                doctors=[
                    DoctorScheduleObservation(
                        doctor_id="d-family-1",
                        name="Dr. Alvarez",
                        specialty="family_medicine",
                        status="available",
                        available_slots=["2026-04-07T09:00"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-family-2",
                        name="Dr. Chen",
                        specialty="family_medicine",
                        status="busy",
                        available_slots=[],
                    ),
                ],
                rooms=[
                    RoomAvailabilityObservation(
                        room_id="room-exam-1",
                        room_type="exam",
                        available_slots=["2026-04-07T09:00"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-exam-2",
                        room_type="exam",
                        available_slots=[],
                    ),
                ],
                info_bank={
                    "p-routine-1": (
                        "Priya needs family medicine. Booking the earliest open slot also clears a "
                        "35 minute waiting-room delay."
                    )
                },
                recommendation_bank={
                    "p-routine-1": PendingRecommendation(
                        recommendation_id="rec-routine-1",
                        patient_id="p-routine-1",
                        command="BookAppointment",
                        doctor_id="d-family-1",
                        room_id="room-exam-1",
                        time_slot="2026-04-07T09:00",
                        explanation="Routine case: book the first available family medicine slot.",
                    )
                },
                grader_name="_grade_task_1",
            ),
            "task_2_multi_patient_triage": TaskScenario(
                task_name="task_2_multi_patient_triage",
                instruction=(
                    "Task 2 (Medium): in a crowded waiting room, escalate the uncertain critical "
                    "chest-pain case for clinician confirmation, route them to the ER, then "
                    "schedule the two stable patients."
                ),
                max_steps=5,
                er_bed_capacity=1,
                patients=[
                    PatientObservation(
                        patient_id="p-critical-1",
                        name="Marcus Reed",
                        age=58,
                        symptoms=["crushing chest pain", "shortness of breath", "oxygen saturation 89%"],
                        acuity="critical",
                        required_specialty="emergency",
                        notes=["Critical. Do not delay escalation."],
                        estimated_wait_minutes=12,
                        uncertainty_level="high",
                        requires_clinician_review=True,
                    ),
                    PatientObservation(
                        patient_id="p-ankle-1",
                        name="Lena Ortiz",
                        age=25,
                        symptoms=["swollen ankle", "pain when bearing weight"],
                        acuity="urgent",
                        required_specialty="orthopedics",
                        notes=["Stable enough for clinic scheduling after triage is clear."],
                        estimated_wait_minutes=40,
                        uncertainty_level="medium",
                    ),
                    PatientObservation(
                        patient_id="p-throat-1",
                        name="Noah Patel",
                        age=17,
                        symptoms=["sore throat", "low-grade fever"],
                        acuity="routine",
                        required_specialty="family_medicine",
                        notes=["Routine same-day family medicine visit is acceptable."],
                        estimated_wait_minutes=55,
                        uncertainty_level="low",
                    ),
                ],
                doctors=[
                    DoctorScheduleObservation(
                        doctor_id="d-er-1",
                        name="Dr. Singh",
                        specialty="emergency",
                        status="available",
                        available_slots=["2026-04-07T10:00"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-ortho-1",
                        name="Dr. Foster",
                        specialty="orthopedics",
                        status="available",
                        available_slots=["2026-04-07T10:30"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-family-2",
                        name="Dr. Nguyen",
                        specialty="family_medicine",
                        status="available",
                        available_slots=["2026-04-07T11:00"],
                    ),
                ],
                rooms=[
                    RoomAvailabilityObservation(
                        room_id="room-er-1",
                        room_type="er",
                        available_slots=["2026-04-07T10:00"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-exam-3",
                        room_type="exam",
                        available_slots=["2026-04-07T10:30"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-exam-4",
                        room_type="exam",
                        available_slots=["2026-04-07T11:00"],
                    ),
                ],
                info_bank={
                    "p-critical-1": (
                        "The chest-pain case is high uncertainty and high risk. Safe workflow: "
                        "escalate for clinician confirmation, then route to the ER immediately."
                    ),
                    "p-ankle-1": "Orthopedics can evaluate at 10:30 in room-exam-3.",
                    "p-throat-1": "Family medicine can evaluate at 11:00 in room-exam-4.",
                },
                recommendation_bank={
                    "p-critical-1": PendingRecommendation(
                        recommendation_id="rec-critical-1",
                        patient_id="p-critical-1",
                        command="SendToER",
                        explanation="Clinician confirms immediate ER transfer for possible myocardial infarction.",
                    ),
                    "p-ankle-1": PendingRecommendation(
                        recommendation_id="rec-ankle-1",
                        patient_id="p-ankle-1",
                        command="BookAppointment",
                        doctor_id="d-ortho-1",
                        room_id="room-exam-3",
                        time_slot="2026-04-07T10:30",
                        explanation="Book orthopedics as the next safest clinic visit.",
                    ),
                    "p-throat-1": PendingRecommendation(
                        recommendation_id="rec-throat-1",
                        patient_id="p-throat-1",
                        command="BookAppointment",
                        doctor_id="d-family-2",
                        room_id="room-exam-4",
                        time_slot="2026-04-07T11:00",
                        explanation="Book family medicine after the critical patient is stabilized.",
                    ),
                },
                grader_name="_grade_task_2",
            ),
            "task_3_specialty_reschedule": TaskScenario(
                task_name="task_3_specialty_reschedule",
                instruction=(
                    "Task 3 (Hard): Dr. Morgan called in sick. Reschedule the four disrupted "
                    "patients with specialty matches, avoid double-booking, and keep waiting "
                    "pressure under control."
                ),
                max_steps=6,
                er_bed_capacity=1,
                patients=[
                    PatientObservation(
                        patient_id="p-cardio-1",
                        name="Amira Khan",
                        age=63,
                        symptoms=["post-stent follow-up"],
                        acuity="urgent",
                        required_specialty="cardiology",
                        notes=["Needs cardiology specifically."],
                        existing_appointment=AppointmentDetails(
                            patient_id="p-cardio-1",
                            doctor_id="d-sick-1",
                            doctor_name="Dr. Morgan",
                            specialty="cardiology",
                            room_id="room-consult-1",
                            time_slot="2026-04-07T13:00",
                            status="canceled",
                        ),
                        estimated_wait_minutes=75,
                        uncertainty_level="medium",
                    ),
                    PatientObservation(
                        patient_id="p-endo-1",
                        name="Ethan Brooks",
                        age=46,
                        symptoms=["insulin regimen review"],
                        acuity="routine",
                        required_specialty="endocrinology",
                        notes=["Needs endocrinology specifically."],
                        existing_appointment=AppointmentDetails(
                            patient_id="p-endo-1",
                            doctor_id="d-sick-1",
                            doctor_name="Dr. Morgan",
                            specialty="endocrinology",
                            room_id="room-consult-1",
                            time_slot="2026-04-07T13:30",
                            status="canceled",
                        ),
                        estimated_wait_minutes=60,
                        uncertainty_level="medium",
                    ),
                    PatientObservation(
                        patient_id="p-derm-1",
                        name="Sofia Ramos",
                        age=29,
                        symptoms=["new medication rash"],
                        acuity="urgent",
                        required_specialty="dermatology",
                        notes=["Needs dermatology specifically."],
                        existing_appointment=AppointmentDetails(
                            patient_id="p-derm-1",
                            doctor_id="d-sick-1",
                            doctor_name="Dr. Morgan",
                            specialty="dermatology",
                            room_id="room-consult-2",
                            time_slot="2026-04-07T14:00",
                            status="canceled",
                        ),
                        estimated_wait_minutes=80,
                        uncertainty_level="medium",
                    ),
                    PatientObservation(
                        patient_id="p-pulm-1",
                        name="Daniel Lee",
                        age=51,
                        symptoms=["asthma control review"],
                        acuity="routine",
                        required_specialty="pulmonology",
                        notes=["Needs pulmonology specifically."],
                        existing_appointment=AppointmentDetails(
                            patient_id="p-pulm-1",
                            doctor_id="d-sick-1",
                            doctor_name="Dr. Morgan",
                            specialty="pulmonology",
                            room_id="room-consult-2",
                            time_slot="2026-04-07T14:30",
                            status="canceled",
                        ),
                        estimated_wait_minutes=65,
                        uncertainty_level="low",
                    ),
                ],
                doctors=[
                    DoctorScheduleObservation(
                        doctor_id="d-sick-1",
                        name="Dr. Morgan",
                        specialty="multi_specialty",
                        status="sick",
                        available_slots=[],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-cardio-2",
                        name="Dr. Ellis",
                        specialty="cardiology",
                        status="available",
                        available_slots=["2026-04-07T13:00"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-endo-2",
                        name="Dr. Iyer",
                        specialty="endocrinology",
                        status="available",
                        available_slots=["2026-04-07T13:30"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-derm-2",
                        name="Dr. Park",
                        specialty="dermatology",
                        status="available",
                        available_slots=["2026-04-07T14:00"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-pulm-2",
                        name="Dr. Wilson",
                        specialty="pulmonology",
                        status="available",
                        available_slots=["2026-04-07T14:30"],
                    ),
                ],
                rooms=[
                    RoomAvailabilityObservation(
                        room_id="room-consult-1",
                        room_type="consult",
                        available_slots=["2026-04-07T13:00", "2026-04-07T13:30"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-consult-2",
                        room_type="consult",
                        available_slots=["2026-04-07T14:00", "2026-04-07T14:30"],
                    ),
                ],
                info_bank={
                    "p-cardio-1": "Cardiology should be rescheduled first to keep urgent waiting pressure down.",
                    "p-endo-1": "Endocrinology can move to 13:30 in room-consult-1.",
                    "p-derm-1": "Dermatology should be scheduled early because the rash is worsening.",
                    "p-pulm-1": "Pulmonology can move to 14:30 in room-consult-2.",
                },
                recommendation_bank={
                    "p-cardio-1": PendingRecommendation(
                        recommendation_id="rec-cardio-1",
                        patient_id="p-cardio-1",
                        command="BookAppointment",
                        doctor_id="d-cardio-2",
                        room_id="room-consult-1",
                        time_slot="2026-04-07T13:00",
                        explanation="Reschedule cardiology first to control urgent wait time.",
                    ),
                    "p-endo-1": PendingRecommendation(
                        recommendation_id="rec-endo-1",
                        patient_id="p-endo-1",
                        command="BookAppointment",
                        doctor_id="d-endo-2",
                        room_id="room-consult-1",
                        time_slot="2026-04-07T13:30",
                        explanation="Endocrinology can follow after the urgent cases are handled.",
                    ),
                    "p-derm-1": PendingRecommendation(
                        recommendation_id="rec-derm-1",
                        patient_id="p-derm-1",
                        command="BookAppointment",
                        doctor_id="d-derm-2",
                        room_id="room-consult-2",
                        time_slot="2026-04-07T14:00",
                        explanation="Schedule dermatology promptly to limit urgent waiting pressure.",
                    ),
                    "p-pulm-1": PendingRecommendation(
                        recommendation_id="rec-pulm-1",
                        patient_id="p-pulm-1",
                        command="BookAppointment",
                        doctor_id="d-pulm-2",
                        room_id="room-consult-2",
                        time_slot="2026-04-07T14:30",
                        explanation="Pulmonology can be placed in the last open specialty slot.",
                    ),
                },
                grader_name="_grade_task_3",
            ),
            "task_4_ambiguous_walk_in": TaskScenario(
                task_name="task_4_ambiguous_walk_in",
                instruction=(
                    "Task 4 (Hard): an ambiguous abdominal-pain walk-in needs more information "
                    "before routing. Clarify the case, then clear the remaining specialty backlog "
                    "without letting wait pressure spike."
                ),
                max_steps=6,
                er_bed_capacity=1,
                patients=[
                    PatientObservation(
                        patient_id="p-abd-1",
                        name="Maya Desai",
                        age=41,
                        symptoms=["right lower quadrant abdominal pain", "nausea", "pain score 7/10"],
                        acuity="urgent",
                        required_specialty="gastroenterology",
                        notes=["Stable vitals, but the safest clinic routing is not obvious without clarification."],
                        estimated_wait_minutes=50,
                        uncertainty_level="high",
                    ),
                    PatientObservation(
                        patient_id="p-diabetes-1",
                        name="Harper Collins",
                        age=52,
                        symptoms=["insulin pump review", "high glucose variability"],
                        acuity="routine",
                        required_specialty="endocrinology",
                        notes=["Needs same-day endocrinology follow-up."],
                        estimated_wait_minutes=65,
                        uncertainty_level="medium",
                    ),
                    PatientObservation(
                        patient_id="p-eczema-1",
                        name="Jordan Kim",
                        age=28,
                        symptoms=["eczema flare", "itching despite topical steroids"],
                        acuity="routine",
                        required_specialty="dermatology",
                        notes=["Needs dermatology, but is stable enough to wait behind the urgent case."],
                        estimated_wait_minutes=45,
                        uncertainty_level="low",
                    ),
                ],
                doctors=[
                    DoctorScheduleObservation(
                        doctor_id="d-gi-1",
                        name="Dr. Romero",
                        specialty="gastroenterology",
                        status="available",
                        available_slots=["2026-04-07T15:00"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-endo-3",
                        name="Dr. Malik",
                        specialty="endocrinology",
                        status="available",
                        available_slots=["2026-04-07T15:30"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-derm-3",
                        name="Dr. Shah",
                        specialty="dermatology",
                        status="available",
                        available_slots=["2026-04-07T16:00"],
                    ),
                ],
                rooms=[
                    RoomAvailabilityObservation(
                        room_id="room-consult-3",
                        room_type="consult",
                        available_slots=["2026-04-07T15:00"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-consult-4",
                        room_type="consult",
                        available_slots=["2026-04-07T15:30"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-exam-5",
                        room_type="exam",
                        available_slots=["2026-04-07T16:00"],
                    ),
                ],
                info_bank={
                    "p-abd-1": (
                        "Ultrasound triage note: vitals are stable, there is no rebound tenderness, "
                        "and same-day gastroenterology is the safest clinic route."
                    ),
                    "p-diabetes-1": "Endocrinology can see Harper at 15:30 in room-consult-4.",
                    "p-eczema-1": "Dermatology can see Jordan at 16:00 in room-exam-5.",
                },
                recommendation_bank={
                    "p-abd-1": PendingRecommendation(
                        recommendation_id="rec-abd-1",
                        patient_id="p-abd-1",
                        command="BookAppointment",
                        doctor_id="d-gi-1",
                        room_id="room-consult-3",
                        time_slot="2026-04-07T15:00",
                        explanation="After clarification, book the abdominal case into the gastroenterology overflow slot.",
                    ),
                    "p-diabetes-1": PendingRecommendation(
                        recommendation_id="rec-diabetes-1",
                        patient_id="p-diabetes-1",
                        command="BookAppointment",
                        doctor_id="d-endo-3",
                        room_id="room-consult-4",
                        time_slot="2026-04-07T15:30",
                        explanation="Move the insulin-review patient into endocrinology once the urgent case is stabilized.",
                    ),
                    "p-eczema-1": PendingRecommendation(
                        recommendation_id="rec-eczema-1",
                        patient_id="p-eczema-1",
                        command="BookAppointment",
                        doctor_id="d-derm-3",
                        room_id="room-exam-5",
                        time_slot="2026-04-07T16:00",
                        explanation="Route the eczema flare to dermatology after the urgent backlog clears.",
                    ),
                },
                grader_name="_grade_task_4",
            ),
            "task_5_evening_surge_coordination": TaskScenario(
                task_name="task_5_evening_surge_coordination",
                instruction=(
                    "Task 5 (Hard): during an evening surge, escalate the septic patient for "
                    "clinician confirmation, clarify the arrhythmia walk-in, and schedule the "
                    "remaining patients without overwhelming capacity."
                ),
                max_steps=7,
                er_bed_capacity=1,
                patients=[
                    PatientObservation(
                        patient_id="p-sepsis-1",
                        name="Elena Cruz",
                        age=67,
                        symptoms=["fever 39.5C", "confusion", "low blood pressure"],
                        acuity="critical",
                        required_specialty="emergency",
                        notes=["Possible sepsis. Must be addressed first in the surge."],
                        estimated_wait_minutes=8,
                        uncertainty_level="high",
                        requires_clinician_review=True,
                    ),
                    PatientObservation(
                        patient_id="p-arrhythmia-1",
                        name="Owen Brooks",
                        age=39,
                        symptoms=["palpitations", "dizziness", "heart rate 118"],
                        acuity="urgent",
                        required_specialty="cardiology",
                        notes=["High uncertainty, but not every arrhythmia walk-in needs the ER."],
                        estimated_wait_minutes=35,
                        uncertainty_level="high",
                    ),
                    PatientObservation(
                        patient_id="p-headache-1",
                        name="Rina Bose",
                        age=33,
                        symptoms=["migraine flare", "light sensitivity", "nausea"],
                        acuity="urgent",
                        required_specialty="neurology",
                        notes=["Neurology can handle this after the septic patient is stabilized."],
                        estimated_wait_minutes=55,
                        uncertainty_level="medium",
                    ),
                    PatientObservation(
                        patient_id="p-followup-1",
                        name="Caleb Wright",
                        age=48,
                        symptoms=["blood pressure follow-up"],
                        acuity="routine",
                        required_specialty="family_medicine",
                        notes=["Routine same-day family medicine follow-up."],
                        estimated_wait_minutes=60,
                        uncertainty_level="low",
                    ),
                ],
                doctors=[
                    DoctorScheduleObservation(
                        doctor_id="d-er-2",
                        name="Dr. Santos",
                        specialty="emergency",
                        status="available",
                        available_slots=["2026-04-07T17:00"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-cardio-3",
                        name="Dr. Freeman",
                        specialty="cardiology",
                        status="available",
                        available_slots=["2026-04-07T17:30"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-neuro-1",
                        name="Dr. Okafor",
                        specialty="neurology",
                        status="available",
                        available_slots=["2026-04-07T18:00"],
                    ),
                    DoctorScheduleObservation(
                        doctor_id="d-family-3",
                        name="Dr. Liu",
                        specialty="family_medicine",
                        status="available",
                        available_slots=["2026-04-07T18:30"],
                    ),
                ],
                rooms=[
                    RoomAvailabilityObservation(
                        room_id="room-er-2",
                        room_type="er",
                        available_slots=["2026-04-07T17:00"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-consult-5",
                        room_type="consult",
                        available_slots=["2026-04-07T17:30"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-consult-6",
                        room_type="consult",
                        available_slots=["2026-04-07T18:00"],
                    ),
                    RoomAvailabilityObservation(
                        room_id="room-exam-6",
                        room_type="exam",
                        available_slots=["2026-04-07T18:30"],
                    ),
                ],
                info_bank={
                    "p-sepsis-1": (
                        "Clinician note: likely sepsis. Immediate ER transfer is the safest operational next step."
                    ),
                    "p-arrhythmia-1": (
                        "Telemetry triage note: blood pressure is stable, there is no syncope, and "
                        "same-day cardiology is safer than using the single ER bed."
                    ),
                    "p-headache-1": "Neurology can evaluate Rina at 18:00 in room-consult-6.",
                    "p-followup-1": "Family medicine can see Caleb at 18:30 in room-exam-6.",
                },
                recommendation_bank={
                    "p-sepsis-1": PendingRecommendation(
                        recommendation_id="rec-sepsis-1",
                        patient_id="p-sepsis-1",
                        command="SendToER",
                        explanation="Clinician confirms immediate ER transfer for suspected sepsis.",
                    ),
                    "p-arrhythmia-1": PendingRecommendation(
                        recommendation_id="rec-arrhythmia-1",
                        patient_id="p-arrhythmia-1",
                        command="BookAppointment",
                        doctor_id="d-cardio-3",
                        room_id="room-consult-5",
                        time_slot="2026-04-07T17:30",
                        explanation="Book cardiology after reviewing the stable telemetry note.",
                    ),
                    "p-headache-1": PendingRecommendation(
                        recommendation_id="rec-headache-1",
                        patient_id="p-headache-1",
                        command="BookAppointment",
                        doctor_id="d-neuro-1",
                        room_id="room-consult-6",
                        time_slot="2026-04-07T18:00",
                        explanation="Schedule neurology after the ER escalation and arrhythmia clarification.",
                    ),
                    "p-followup-1": PendingRecommendation(
                        recommendation_id="rec-followup-1",
                        patient_id="p-followup-1",
                        command="BookAppointment",
                        doctor_id="d-family-3",
                        room_id="room-exam-6",
                        time_slot="2026-04-07T18:30",
                        explanation="Use the remaining family medicine slot for the routine follow-up.",
                    ),
                },
                grader_name="_grade_task_5",
            ),
        }
