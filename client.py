# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hospital Triage environment client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import HospitalTriageAction, HospitalTriageObservation, HospitalTriageState


class HospitalTriageEnv(
    EnvClient[HospitalTriageAction, HospitalTriageObservation, HospitalTriageState]
):
    """
    Client for the Hospital Triage Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with HospitalTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(HospitalTriageAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = HospitalTriageEnv.from_docker_image("hospital_triage-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(HospitalTriageAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HospitalTriageAction) -> Dict:
        """
        Convert HospitalTriageAction to JSON payload for step message.

        Args:
            action: HospitalTriageAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[HospitalTriageObservation]:
        """
        Parse server response into StepResult[HospitalTriageObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HospitalTriageObservation
        """
        obs_data: Dict[str, Any] = dict(payload.get("observation", {}))
        obs_data.setdefault("done", payload.get("done", False))
        obs_data.setdefault("reward", payload.get("reward"))
        observation = HospitalTriageObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> HospitalTriageState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return HospitalTriageState.model_validate(payload)
