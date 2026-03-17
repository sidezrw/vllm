# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def _env_flag(name: str) -> bool:
    return bool(int(os.getenv(name, "0")))


def _get_backend_name() -> str:
    if _env_flag("VLLM_IMAGE_DECODE_CACHE_ENABLED"):
        return "null_decode"

    return os.getenv("VLLM_IMAGE_LOADER_BACKEND", "unknown")


def _count_images(mm_features: Any) -> tuple[bool, int]:
    if not mm_features:
        return False, 0

    image_count = sum(
        1 for feature in mm_features if getattr(feature, "modality", None) == "image"
    )
    return image_count > 0, image_count


def _count_image_requests(requests: list[Any]) -> tuple[int, int]:
    image_req_count = 0
    total_image_count = 0
    for request in requests:
        has_image, image_count = _count_images(getattr(request, "mm_features", None))
        if has_image:
            image_req_count += 1
            total_image_count += image_count

    return image_req_count, total_image_count


def _count_scheduled_encoder_inputs(
    scheduled_encoder_inputs: dict[str, list[int]],
) -> int:
    return sum(len(input_ids) for input_ids in scheduled_encoder_inputs.values())


class _TraceWriter:
    def __init__(self, role: str, enabled_env: str) -> None:
        self.role = role
        self.enabled = _env_flag(enabled_env)
        self.backend = _get_backend_name()
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._iteration = 0
        self._file = None

        if not self.enabled:
            return

        trace_dir = Path(os.getenv("VLLM_BATCH_PACKING_TRACE_DIR", "."))
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"batch_packing_{role}_{self.pid}.jsonl"
            self._file = trace_path.open("a", encoding="utf-8", buffering=1024 * 1024)
            atexit.register(self.close)
        except OSError:
            self.enabled = False
            self._file = None

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def next_iteration_id(self) -> int:
        self._iteration += 1
        return self._iteration

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled or self._file is None:
            return

        payload = {
            "event": event,
            "ts_monotonic_ns": time.monotonic_ns(),
            "ts_wall_ns": time.time_ns(),
            "pid": self.pid,
            "role": self.role,
            "backend": self.backend,
        }
        payload.update({k: v for k, v in fields.items() if v is not None})

        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            self._file.write(f"{line}\n")


_request_ready_trace = _TraceWriter(
    role="frontend",
    enabled_env="VLLM_BATCH_PACKING_TRACE_REQUEST_READY",
)
_schedule_batch_trace = _TraceWriter(
    role="scheduler",
    enabled_env="VLLM_BATCH_PACKING_TRACE_SCHEDULE_BATCH",
)
_execute_batch_trace = _TraceWriter(
    role="worker",
    enabled_env="VLLM_BATCH_PACKING_TRACE_EXECUTE_BATCH",
)


def trace_request_ready(request: Any) -> None:
    if not _request_ready_trace.enabled:
        return

    has_image, image_count = _count_images(getattr(request, "mm_features", None))
    _request_ready_trace.emit(
        "request_ready",
        request_id=request.request_id,
        has_image=has_image,
        image_count=image_count,
    )


def trace_schedule_batch(
    requests: list[Any],
    scheduled_tokens: int,
    scheduled_encoder_inputs: dict[str, list[int]],
    waiting_size: int,
    running_size: int,
) -> None:
    if not _schedule_batch_trace.enabled:
        return

    image_req_count, total_image_count = _count_image_requests(requests)
    _schedule_batch_trace.emit(
        "schedule_batch",
        iteration_id=_schedule_batch_trace.next_iteration_id(),
        total_reqs=len(requests),
        image_req_count=image_req_count,
        total_image_count=total_image_count,
        scheduled_tokens=scheduled_tokens,
        scheduled_encoder_input_count=_count_scheduled_encoder_inputs(
            scheduled_encoder_inputs
        ),
        waiting_size=waiting_size,
        running_size=running_size,
    )


def trace_execute_batch(
    req_ids: list[str],
    requests_by_id: dict[str, Any],
    scheduled_tokens: int,
    scheduled_encoder_inputs: dict[str, list[int]],
) -> None:
    if not _execute_batch_trace.enabled:
        return

    requests = [requests_by_id[req_id] for req_id in req_ids]
    image_req_count, total_image_count = _count_image_requests(requests)
    _execute_batch_trace.emit(
        "execute_batch",
        iteration_id=_execute_batch_trace.next_iteration_id(),
        total_reqs=len(req_ids),
        image_req_count=image_req_count,
        total_image_count=total_image_count,
        scheduled_tokens=scheduled_tokens,
        scheduled_encoder_input_count=_count_scheduled_encoder_inputs(
            scheduled_encoder_inputs
        ),
    )
