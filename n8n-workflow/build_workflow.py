#!/usr/bin/env python3
"""Build the importable n8n workflow for the rednote-bot worker API.

The workflow intentionally does not use an n8n Telegram Trigger.  The
existing Telegram bot remains the only Telegram webhook owner; in n8n mode it
creates the status message, stores the user's selected settings in the worker
job, and POSTs that job to this workflow's Webhook node.

Required n8n environment variables:
- WORKER_BASE_URL   e.g. https://rednote-bot-gzd0.onrender.com
- WORKER_API_KEY    same value configured on the Render worker
- N8N_WEBHOOK_SECRET (optional, but strongly recommended)

For self-hosted n8n, expression access to process environment variables may
also need N8N_BLOCK_ENV_ACCESS_IN_NODE=false.

Run:
    python3 build_workflow.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

NAMESPACE = uuid.UUID("9a08f0e5-e72a-4ea4-9d84-8ae196769f84")
WORKFLOW_NAME = "Rednote Bot — Whisper First Person Recap (Worker API)"
STAGES = (
    ("download", "Download", 10),
    ("transcribe", "Transcribe", 15),
    ("rewrite", "Rewrite", 15),
    ("tts", "TTS", 10),
    ("render", "Render", 15),
    ("deliver", "Deliver", 10),
)


def _node_id(name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, name))


def _node(
    name: str,
    node_type: str,
    type_version: float,
    parameters: dict[str, Any],
    position: tuple[int, int],
    *,
    retry: bool = False,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "parameters": parameters,
        "id": _node_id(name),
        "name": name,
        "type": node_type,
        "typeVersion": type_version,
        "position": list(position),
    }
    if retry:
        node.update(
            {
                "retryOnFail": True,
                "maxTries": 3,
                "waitBetweenTries": 5000,
            }
        )
    return node


def _worker_url(path_expression: str) -> str:
    return (
        "={{ ($env.WORKER_BASE_URL || '').replace(/\\/+$/, '') + "
        f"{path_expression} }}}}"
    )


def _http_node(
    name: str,
    method: str,
    path_expression: str,
    position: tuple[int, int],
    *,
    json_body: str | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "method": method,
        "url": _worker_url(path_expression),
        "sendHeaders": True,
        "headerParameters": {
            "parameters": [
                {
                    "name": "X-Worker-Key",
                    "value": "={{ $env.WORKER_API_KEY || '' }}",
                },
                {"name": "Content-Type", "value": "application/json"},
            ]
        },
        "options": {"timeout": 120000},
    }
    if json_body is not None:
        parameters.update(
            {
                "sendBody": True,
                "contentType": "json",
                "specifyBody": "json",
                "jsonBody": json_body,
            }
        )
    return _node(
        name,
        "n8n-nodes-base.httpRequest",
        4.2,
        parameters,
        position,
        retry=True,
    )


def _code_node(name: str, js_code: str, position: tuple[int, int]) -> dict[str, Any]:
    return _node(
        name,
        "n8n-nodes-base.code",
        2,
        {"mode": "runOnceForAllItems", "jsCode": js_code.strip()},
        position,
    )


def _boolean_if_node(name: str, field: str, position: tuple[int, int]) -> dict[str, Any]:
    condition_id = str(uuid.uuid5(NAMESPACE, f"condition:{name}"))
    return _node(
        name,
        "n8n-nodes-base.if",
        2,
        {
            "conditions": {
                "options": {
                    "caseSensitive": True,
                    "leftValue": "",
                    "typeValidation": "loose",
                    "version": 2,
                },
                "conditions": [
                    {
                        "id": condition_id,
                        "name": condition_id,
                        "leftValue": f"={{{{ $json.{field} }}}}",
                        "rightValue": True,
                        "operator": {
                            "type": "boolean",
                            "operation": "true",
                            "singleValue": True,
                        },
                    }
                ],
                "combinator": "and",
            },
            "options": {},
        },
        position,
    )


def _target(node_name: str) -> dict[str, Any]:
    return {"node": node_name, "type": "main", "index": 0}


PAYLOAD_CODE = r"""
const root = ($input.first() && $input.first().json) || {};
const data = root.body && typeof root.body === 'object' ? root.body : root;
const headers = root.headers || data.headers || {};
const lowerHeaders = {};
for (const [key, value] of Object.entries(headers)) {
  lowerHeaders[String(key).toLowerCase()] = value;
}

const expectedSecret = String($env.N8N_WEBHOOK_SECRET || '');
const suppliedSecret = String(lowerHeaders['x-n8n-secret'] || '');
if (expectedSecret && suppliedSecret !== expectedSecret) {
  throw new Error('Invalid X-N8N-Secret header');
}

const url = String(data.url || '').trim();
if (!/^https?:\/\//i.test(url)) {
  throw new Error('A valid http/https video url is required');
}

const chatId = Number(data.chat_id);
if (!Number.isFinite(chatId) || chatId === 0) {
  throw new Error('A valid chat_id is required');
}

const generatedJobId =
  'j' + Date.now().toString(36) +
  Math.random().toString(36).slice(2, 10);
const jobId = String(data.job_id || generatedJobId).trim();
if (!/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
  throw new Error('Invalid job_id');
}

const optionalNumber = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const payload = {
  job_id: jobId,
  chat_id: chatId,
  reply_msg_id: optionalNumber(data.reply_msg_id),
  status_message_id: optionalNumber(data.status_message_id),
  url,
  create_status_message:
    data.create_status_message === undefined
      ? !optionalNumber(data.status_message_id)
      : Boolean(data.create_status_message),
};

if (data.settings && typeof data.settings === 'object' && !Array.isArray(data.settings)) {
  payload.settings = data.settings;
}

return [{ json: payload }];
"""


STAGE_STATUS_CODE = r"""
const data = ($input.first() && $input.first().json) || {};
const info = ((data.stages || {})['__STAGE__']) || {};
const stageStatus = String(info.status || '').toLowerCase();
const jobStatus = String(data.status || '').toLowerCase();

return [{
  json: {
    ...data,
    stage: '__STAGE__',
    stage_status: stageStatus,
    stage_done: stageStatus === 'done',
    stage_failed:
      stageStatus === 'failed' ||
      stageStatus === 'cancelled' ||
      jobStatus === 'failed' ||
      jobStatus === 'cancelled',
    stage_error: String(info.error || data.error || ''),
  },
}];
"""


ERROR_CODE = r"""
const data = ($input.first() && $input.first().json) || {};
const reason = String(
  data.stage_error ||
  data.error ||
  `${data.stage || 'worker'} stage failed`
).slice(0, 1000);

return [{
  json: {
    job_id: data.job_id,
    stage: data.stage || '',
    stage_status: data.stage_status || data.status || 'failed',
    reason,
  },
}];
"""


SUCCESS_CODE = r"""
const data = ($input.first() && $input.first().json) || {};
return [{
  json: {
    ok: true,
    job_id: data.job_id || '',
    deleted: data.deleted !== false,
    message: `Worker job completed and cleaned up: ${data.job_id || ''}`,
  },
}];
"""


STICKY_CONTENT = """## Setup မှတ်ချက်

ဒီ workflow မှာ **Telegram Trigger မသုံးပါ**။ လက်ရှိ Telegram bot ကပဲ Telegram webhook/menu/settings/status ကို ကိုင်တွယ်ပြီး ဒီ Webhook ဆီ job ပို့ပါမယ်။

n8n environment မှာ အောက်ပါတို့ ထည့်ပါ:

- `WORKER_BASE_URL=https://your-render-service.onrender.com`
- `WORKER_API_KEY=<Render မှာ ထည့်ထားတဲ့ တူညီသောလျှို့ဝှက်ကုဒ်>`
- `N8N_WEBHOOK_SECRET=<Render မှာ ထည့်ထားတဲ့ တူညီသောလျှို့ဝှက်ကုဒ်>`

Self-hosted n8n ဆိုလျှင် expression တွေက env ဖတ်နိုင်ရန် `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` လည်း လိုနိုင်ပါတယ်။

စစ်ဆေးပြီးနောက် workflow ကို **Activate** လုပ်ပါ။ Production webhook URL ကို Render `N8N_WEBHOOK_URL` ထဲ ထည့်ပြီးမှ `PIPELINE_MODE=n8n` ပြောင်းပါ။

အသေးစိတ်ကို repository ထဲက `n8n-workflow/README.md` မှာ ကြည့်ပါ။
"""


def build_workflow() -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    connections: dict[str, Any] = {}

    sticky_name = "Setup မှတ်ချက်"
    webhook_name = "Telegram Bot မှလာသော Job"
    payload_name = "Payload စစ်"
    create_name = "Worker Job ဖန်တီး/အတည်ပြု"

    nodes.append(
        _node(
            sticky_name,
            "n8n-nodes-base.stickyNote",
            1,
            {
                "content": STICKY_CONTENT,
                "height": 560,
                "width": 720,
                "color": 4,
            },
            (-820, -40),
        )
    )
    nodes.append(
        _node(
            webhook_name,
            "n8n-nodes-base.webhook",
            2,
            {
                "httpMethod": "POST",
                "path": "rednote-recap-job",
                "responseMode": "onReceived",
                "options": {},
            },
            (-40, 300),
        )
    )
    nodes.append(_code_node(payload_name, PAYLOAD_CODE, (240, 300)))
    nodes.append(
        _http_node(
            create_name,
            "POST",
            "'/api/v1/jobs'",
            (520, 300),
            json_body="={{ JSON.stringify($json) }}",
        )
    )
    connections[webhook_name] = {"main": [[_target(payload_name)]]}
    connections[payload_name] = {"main": [[_target(create_name)]]}

    error_code_name = "အမှားအချက်အလက် ပြင်"
    error_fail_name = "Worker ကို Failed မှတ်"
    error_cleanup_name = "အမှားဖြစ်လျှင် Job ရှင်း"
    stop_name = "Workflow Error အဖြစ် ရပ်"
    success_cleanup_name = "ပြီးလျှင် Job ရှင်း"
    success_name = "ပြီးဆုံးကြောင်း မှတ်"

    first_stage_start = ""
    previous_done_if = ""

    stage_block_width = 1320
    first_x = 820
    for index, (stage, label, wait_seconds) in enumerate(STAGES):
        x = first_x + index * stage_block_width
        start_name = f"{label} စတင်"
        wait_name = f"{label} စောင့်"
        status_name = f"{label} status"
        status_code_name = f"{label} status စစ်"
        done_if_name = f"{label} ပြီးပြီလား"
        failed_if_name = f"{label} ပျက်လား"

        if index == 0:
            first_stage_start = start_name
            connections[create_name] = {"main": [[_target(start_name)]]}
        else:
            connections[previous_done_if] = {
                "main": [[_target(start_name)], [_target(failed_if_name)]]
            }

        nodes.append(
            _http_node(
                start_name,
                "POST",
                (
                    "'/api/v1/jobs/' + encodeURIComponent($json.job_id) + "
                    f"'/stages/{stage}'"
                ),
                (x, 300),
                json_body="={}",
            )
        )
        nodes.append(
            _node(
                wait_name,
                "n8n-nodes-base.wait",
                1.1,
                {"amount": wait_seconds, "unit": "seconds"},
                (x + 280, 300),
            )
        )
        nodes.append(
            _http_node(
                status_name,
                "GET",
                "'/api/v1/jobs/' + encodeURIComponent($json.job_id)",
                (x + 560, 300),
            )
        )
        nodes.append(
            _code_node(
                status_code_name,
                STAGE_STATUS_CODE.replace("__STAGE__", stage),
                (x + 840, 300),
            )
        )
        nodes.append(_boolean_if_node(done_if_name, "stage_done", (x + 1120, 300)))
        nodes.append(_boolean_if_node(failed_if_name, "stage_failed", (x + 1120, 540)))

        connections[start_name] = {"main": [[_target(wait_name)]]}
        connections[wait_name] = {"main": [[_target(status_name)]]}
        connections[status_name] = {"main": [[_target(status_code_name)]]}
        connections[status_code_name] = {"main": [[_target(done_if_name)]]}
        connections[failed_if_name] = {
            "main": [[_target(error_code_name)], [_target(wait_name)]]
        }
        previous_done_if = done_if_name
        # Each done IF needs its own false branch.  It is overwritten above
        # only for use as the next stage's true-branch carrier, so store the
        # false target separately by rebuilding it after the next iteration.
        # To keep this generator simple and deterministic, set it now; the
        # next iteration replaces the entry with both branches explicitly.
        connections[done_if_name] = {
            "main": [[], [_target(failed_if_name)]]
        }

    # Replace each intermediate done-IF placeholder with its real next-stage
    # true branch while preserving its own false branch.
    for index, (_stage, label, _wait) in enumerate(STAGES[:-1]):
        done_name = f"{label} ပြီးပြီလား"
        failed_name = f"{label} ပျက်လား"
        next_label = STAGES[index + 1][1]
        connections[done_name] = {
            "main": [
                [_target(f"{next_label} စတင်")],
                [_target(failed_name)],
            ]
        }

    last_done_if = f"{STAGES[-1][1]} ပြီးပြီလား"
    last_failed_if = f"{STAGES[-1][1]} ပျက်လား"
    final_x = first_x + len(STAGES) * stage_block_width

    nodes.append(
        _http_node(
            success_cleanup_name,
            "DELETE",
            "'/api/v1/jobs/' + encodeURIComponent($json.job_id)",
            (final_x, 300),
        )
    )
    nodes.append(_code_node(success_name, SUCCESS_CODE, (final_x + 300, 300)))
    connections[last_done_if] = {
        "main": [[_target(success_cleanup_name)], [_target(last_failed_if)]]
    }
    connections[success_cleanup_name] = {"main": [[_target(success_name)]]}

    nodes.append(_code_node(error_code_name, ERROR_CODE, (820, 860)))
    nodes.append(
        _http_node(
            error_fail_name,
            "POST",
            "'/api/v1/jobs/' + encodeURIComponent($json.job_id) + '/fail'",
            (1100, 860),
            json_body="={{ JSON.stringify({ reason: $json.reason }) }}",
        )
    )
    nodes.append(
        _http_node(
            error_cleanup_name,
            "DELETE",
            "'/api/v1/jobs/' + encodeURIComponent($json.job_id)",
            (1380, 860),
        )
    )
    nodes.append(
        _node(
            stop_name,
            "n8n-nodes-base.stopAndError",
            1,
            {
                "errorMessage": (
                    "={{ 'Worker job failed and was cleaned up: ' + "
                    "($json.job_id || '') }}"
                )
            },
            (1660, 860),
        )
    )
    connections[error_code_name] = {"main": [[_target(error_fail_name)]]}
    connections[error_fail_name] = {"main": [[_target(error_cleanup_name)]]}
    connections[error_cleanup_name] = {"main": [[_target(stop_name)]]}

    # Keep the variable referenced for readers/debuggers even though the
    # explicit connection was added when the first stage was created.
    assert first_stage_start == f"{STAGES[0][1]} စတင်"

    workflow = {
        "name": WORKFLOW_NAME,
        "nodes": nodes,
        "connections": connections,
        "active": False,
        "settings": {
            "saveExecutionProgress": True,
            "executionOrder": "v1",
            "timezone": "Asia/Yangon",
        },
        "staticData": None,
        "pinData": {},
        "meta": {"templateCredsSetupCompleted": False},
    }
    validate_workflow(workflow)
    return workflow


def validate_workflow(workflow: dict[str, Any]) -> None:
    nodes = workflow.get("nodes") or []
    names = [str(node.get("name") or "") for node in nodes]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate node names: {duplicates}")

    name_set = set(names)
    for source, data in (workflow.get("connections") or {}).items():
        if source not in name_set:
            raise ValueError(f"connection source does not exist: {source}")
        for output in (data.get("main") or []):
            for target in output or []:
                if target.get("node") not in name_set:
                    raise ValueError(
                        f"connection target does not exist: {target.get('node')}"
                    )

    serialized = json.dumps(workflow, ensure_ascii=False)
    for stage, _label, _wait in STAGES:
        if f"/stages/{stage}" not in serialized:
            raise ValueError(f"stage start is missing from workflow: {stage}")

    webhook_nodes = [
        node for node in nodes if node.get("type") == "n8n-nodes-base.webhook"
    ]
    if len(webhook_nodes) != 1:
        raise ValueError("workflow must contain exactly one Webhook node")
    webhook_params = webhook_nodes[0].get("parameters") or {}
    if webhook_params.get("responseMode") != "onReceived":
        raise ValueError("Webhook must respond immediately with onReceived")

    for env_name in ("WORKER_BASE_URL", "WORKER_API_KEY", "N8N_WEBHOOK_SECRET"):
        if env_name not in serialized:
            raise ValueError(f"workflow does not reference ${env_name}")


def main() -> None:
    workflow = build_workflow()
    output = Path(__file__).with_name("rednote-n8n-workflow.json")
    output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output} ({len(workflow['nodes'])} nodes)")


if __name__ == "__main__":
    main()
