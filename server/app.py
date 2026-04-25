# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Hospital Triage Environment.

This module creates an HTTP server that exposes the HospitalTriageEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import argparse
import json
import os
from typing import Any

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import HospitalTriageAction, HospitalTriageObservation
    from .hospital_triage_environment import TASK_SEQUENCE, HospitalTriageEnvironment
except ImportError:
    from models import HospitalTriageAction, HospitalTriageObservation
    from server.hospital_triage_environment import TASK_SEQUENCE, HospitalTriageEnvironment


TASK_DEMO_ACTIONS: dict[str, list[dict[str, str]]] = {
    "task_1_routine_checkup": [
        {
            "label": "Step 1",
            "expected_reward": "0.99",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-routine-1",
                    "doctor_id": "d-family-1",
                    "room_id": "room-exam-1",
                    "time_slot": "2026-04-07T09:00",
                },
                indent=2,
            ),
        }
    ],
    "task_2_multi_patient_triage": [
        {
            "label": "Step 1",
            "expected_reward": "0.21",
            "action": json.dumps(
                {
                    "command": "EscalateToClinician",
                    "patient_id": "p-critical-1",
                    "question": "Please confirm the safest next step for this high-risk case.",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 2",
            "expected_reward": "0.60",
            "action": json.dumps(
                {
                    "command": "ConfirmRecommendation",
                    "recommendation_id": "rec-critical-1",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 3",
            "expected_reward": "0.79",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-ankle-1",
                    "doctor_id": "d-ortho-1",
                    "room_id": "room-exam-3",
                    "time_slot": "2026-04-07T10:30",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 4",
            "expected_reward": "0.99",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-throat-1",
                    "doctor_id": "d-family-2",
                    "room_id": "room-exam-4",
                    "time_slot": "2026-04-07T11:00",
                },
                indent=2,
            ),
        },
    ],
    "task_3_specialty_reschedule": [
        {
            "label": "Step 1",
            "expected_reward": "0.21",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-derm-1",
                    "doctor_id": "d-derm-2",
                    "room_id": "room-consult-2",
                    "time_slot": "2026-04-07T14:00",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 2",
            "expected_reward": "0.50",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-cardio-1",
                    "doctor_id": "d-cardio-2",
                    "room_id": "room-consult-1",
                    "time_slot": "2026-04-07T13:00",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 3",
            "expected_reward": "0.70",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-pulm-1",
                    "doctor_id": "d-pulm-2",
                    "room_id": "room-consult-2",
                    "time_slot": "2026-04-07T14:30",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 4",
            "expected_reward": "0.99",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-endo-1",
                    "doctor_id": "d-endo-2",
                    "room_id": "room-consult-1",
                    "time_slot": "2026-04-07T13:30",
                },
                indent=2,
            ),
        },
    ],
    "task_4_ambiguous_walk_in": [
        {
            "label": "Step 1",
            "expected_reward": "0.21",
            "action": json.dumps(
                {
                    "command": "RequestMoreInfo",
                    "patient_id": "p-abd-1",
                    "question": "Please clarify the safest routing decision for this high-uncertainty case.",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 2",
            "expected_reward": "0.50",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-abd-1",
                    "doctor_id": "d-gi-1",
                    "room_id": "room-consult-3",
                    "time_slot": "2026-04-07T15:00",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 3",
            "expected_reward": "0.70",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-diabetes-1",
                    "doctor_id": "d-endo-3",
                    "room_id": "room-consult-4",
                    "time_slot": "2026-04-07T15:30",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 4",
            "expected_reward": "0.99",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-eczema-1",
                    "doctor_id": "d-derm-3",
                    "room_id": "room-exam-5",
                    "time_slot": "2026-04-07T16:00",
                },
                indent=2,
            ),
        },
    ],
    "task_5_evening_surge_coordination": [
        {
            "label": "Step 1",
            "expected_reward": "0.16",
            "action": json.dumps(
                {
                    "command": "EscalateToClinician",
                    "patient_id": "p-sepsis-1",
                    "question": "Please confirm the safest next step for this high-risk case.",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 2",
            "expected_reward": "0.50",
            "action": json.dumps(
                {
                    "command": "ConfirmRecommendation",
                    "recommendation_id": "rec-sepsis-1",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 3",
            "expected_reward": "0.65",
            "action": json.dumps(
                {
                    "command": "RequestMoreInfo",
                    "patient_id": "p-arrhythmia-1",
                    "question": "Please clarify the safest routing decision for this high-uncertainty case.",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 4",
            "expected_reward": "0.74",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-headache-1",
                    "doctor_id": "d-neuro-1",
                    "room_id": "room-consult-6",
                    "time_slot": "2026-04-07T18:00",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 5",
            "expected_reward": "0.89",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-arrhythmia-1",
                    "doctor_id": "d-cardio-3",
                    "room_id": "room-consult-5",
                    "time_slot": "2026-04-07T17:30",
                },
                indent=2,
            ),
        },
        {
            "label": "Step 6",
            "expected_reward": "0.99",
            "action": json.dumps(
                {
                    "command": "BookAppointment",
                    "patient_id": "p-followup-1",
                    "doctor_id": "d-family-3",
                    "room_id": "room-exam-6",
                    "time_slot": "2026-04-07T18:30",
                },
                indent=2,
            ),
        },
    ],
}


def _format_task_reset(data: dict[str, Any]) -> str:
    observation = data.get("observation", {})
    pending_patients = observation.get("pending_patient_ids", [])
    pending_text = ", ".join(pending_patients) if pending_patients else "none"
    return (
        f"## {observation.get('task_name', 'task_unknown')}\n\n"
        f"{observation.get('instruction', '')}\n\n"
        f"- Reward: {float(data.get('reward') or 0.0):.2f}\n"
        f"- Done: {str(bool(data.get('done', False))).lower()}\n"
        f"- Pending patients: {pending_text}"
    )


def _format_demo_steps(task_name: str | None) -> str:
    if not task_name:
        return "Select a task to see a working demo script."

    demo_steps = TASK_DEMO_ACTIONS.get(task_name, [])
    if not demo_steps:
        return "No demo steps are configured for this task yet."

    reward_path = " -> ".join(["0.01"] + [step["expected_reward"] for step in demo_steps])
    blocks = [
        "### Demo Steps",
        "",
        "Reward only moves after you use the Playground tab and click `Step` with a valid action.",
        "",
        f"Expected reward path: `{reward_path}`",
        "",
        "After you load this task:",
        "- switch to `Playground`",
        "- do not click `Reset` there",
        "- optionally click `Get state` once to confirm the `task_name`",
        "",
    ]
    for step in demo_steps:
        blocks.extend(
            [
                f"**{step['label']}**  Expected reward: `{step['expected_reward']}`",
                "",
                "```json",
                step["action"],
                "```",
                "",
            ]
        )
    return "\n".join(blocks).strip()


def build_task_selector_tab(
    web_manager: Any,
    action_fields: list[dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str | None,
):
    del action_fields, is_chat_env, metadata, title, quick_start_md

    import gradio as gr

    env = web_manager.env
    task_catalog = env.task_catalog() if hasattr(env, "task_catalog") else []
    task_map = {item["task_name"]: item for item in task_catalog}
    task_choices = [
        (f"{item['label']} ({item['difficulty']})", item["task_name"])
        for item in task_catalog
    ]
    default_task = TASK_SEQUENCE[0] if task_choices else None

    def describe_task(task_name: str) -> str:
        item = task_map.get(task_name)
        if item is None:
            return "Select a task to see its scenario summary."
        return (
            f"### {item['label']}\n\n"
            f"{item['instruction']}\n\n"
            f"- Difficulty: {item['difficulty']}\n"
            f"- Max steps: {item['max_steps']}\n\n"
            "Load a task here, then use the Playground tab to step through it manually."
        )

    async def reset_selected_task(task_name: str):
        if not task_name:
            return "", "", "", "", "Select a task before resetting."
        try:
            data = await web_manager.reset_environment({"task_name": task_name})
            return (
                describe_task(task_name),
                _format_demo_steps(task_name),
                _format_task_reset(data),
                json.dumps(data, indent=2),
                (
                    f"Loaded {task_name}. Switch to Playground to act on this scenario. "
                    "Do not press Playground Reset unless you want to go back to the default case."
                ),
            )
        except Exception as exc:
            return describe_task(task_name), _format_demo_steps(task_name), "", "", f"Error: {exc}"

    with gr.Blocks(title="Task Selector") as demo:
        gr.Markdown(
            "## Task Selector\n\nChoose which benchmark scenario to load, then follow the demo steps below in the Playground tab."
        )
        task_dropdown = gr.Dropdown(
            choices=task_choices,
            value=default_task,
            label="Benchmark task",
            allow_custom_value=False,
        )
        task_summary = gr.Markdown(value=describe_task(default_task) if default_task else "")
        demo_steps = gr.Markdown(value=_format_demo_steps(default_task))
        with gr.Row():
            load_button = gr.Button("Load selected task", variant="primary")
        task_preview = gr.Markdown(value="Load a task to preview the reset observation.")
        raw_json = gr.Code(label="Reset response", language="json", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)

        task_dropdown.change(
            fn=lambda task_name: (describe_task(task_name), _format_demo_steps(task_name)),
            inputs=[task_dropdown],
            outputs=[task_summary, demo_steps],
        )
        load_button.click(
            fn=reset_selected_task,
            inputs=[task_dropdown],
            outputs=[task_summary, demo_steps, task_preview, raw_json, status],
        )

    return demo


# Create the app with web interface and README integration
app = create_app(
    HospitalTriageEnvironment,
    HospitalTriageAction,
    HospitalTriageObservation,
    env_name="hospital_triage",
    max_concurrent_envs=8,
    gradio_builder=build_task_selector_tab,
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/web")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m hospital_triage.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn hospital_triage.server.app:app --workers 4
    """
    parser = argparse.ArgumentParser(description="Run the Hospital Triage FastAPI app.")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    resolved_host = args.host
    resolved_port = int(os.getenv("PORT", str(args.port)))

    import uvicorn

    uvicorn.run(app, host=resolved_host, port=resolved_port)


if __name__ == "__main__":
    main()
