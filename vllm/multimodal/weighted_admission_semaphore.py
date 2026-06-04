"""Weighted semaphores for bounded GPU decode/preprocess residency."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import threading
import time
import weakref
from typing import Any, Iterable

TOKEN_ATTR = "_vllm_weighted_admission_token"
TOKENS_ATTR = "_vllm_weighted_admission_tokens"
DECODE_TOKEN_ATTR = "_vllm_weighted_decode_token"
_MB = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _mb_env(name: str, default: int) -> int:
    value = _env_int(name, default)
    return max(1, value if value > 0 else default) * _MB


def _nbytes(obj: Any) -> int:
    try:
        return int(obj.numel()) * int(obj.element_size())
    except Exception:
        return 0


class _Reservation:
    def __init__(self, semaphore: "_WeightedSemaphore", weight_bytes: int) -> None:
        self._semaphore = semaphore
        self.weight_bytes = weight_bytes
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            self._released = True
        self._semaphore.release(self)
        return True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.release()


class _WeightedSemaphore:
    def __init__(self, env: str, default_mb: int) -> None:
        self._cv = threading.Condition()
        self._name = env
        self._budget = _mb_env(env, default_mb)
        self._reserved = self._peak = self._acquires = self._releases = 0
        self._inflight = 0

    def acquire(self, weight_bytes: int, label: str) -> _Reservation:
        weight_bytes = max(1, min(int(weight_bytes), self._budget))
        next_log = 0.0
        with self._cv:
            while self._reserved + weight_bytes > self._budget:
                if _env_int("VLLM_GPU_PREPROCESS_TRACE", 0):
                    now = time.monotonic()
                    if now >= next_log:
                        print(
                            "[weighted_admission_semaphore] wait "
                            f"name={self._name} label={label} "
                            f"needed={weight_bytes} "
                            f"reserved={self._reserved} budget={self._budget}",
                            flush=True,
                        )
                        next_log = now + 5.0
                self._cv.wait(0.05)
            self._reserved += weight_bytes
            self._peak = max(self._peak, self._reserved)
            self._acquires += 1
            self._inflight += 1
        return _Reservation(self, weight_bytes)

    def release(self, reservation: _Reservation) -> None:
        with self._cv:
            self._reserved = max(0, self._reserved - reservation.weight_bytes)
            self._releases += 1
            self._inflight = max(0, self._inflight - 1)
            self._cv.notify_all()

    def stats(self) -> dict[str, int]:
        with self._cv:
            return {
                "budget_bytes": self._budget,
                "reserved_bytes": self._reserved,
                "peak_reserved_bytes": self._peak,
                "acquires": self._acquires,
                "releases": self._releases,
                "inflight": self._inflight,
            }


_PREPROCESS_SEMAPHORE = _WeightedSemaphore("VLLM_GPU_PREPROCESS_BUDGET_MB", 8192)
_RESIDENT_SEMAPHORE = _WeightedSemaphore(
    "VLLM_GPU_PREPROCESS_RESIDENT_BUDGET_MB",
    _env_int("VLLM_GPU_PREPROCESS_BUDGET_MB", 8192),
)
_DECODE_SEMAPHORE = _WeightedSemaphore("VLLM_GPU_DECODE_BUDGET_MB", 1536)
_DECODE_RESIDENCY_SEMAPHORE = _WeightedSemaphore("VLLM_GPU_DECODE_RESIDENCY_BUDGET_MB", 512)
_REQUEST_LOCK = threading.Lock()
_REQUEST_TOKENS: dict[str, list[_Reservation]] = {}
_FINISHED_EARLY: dict[str, None] = {}


def _preprocess_floor() -> int:
    return _mb_env(
        "VLLM_GPU_PREPROCESS_RESERVATION_MB",
        _env_int("VLLM_GPU_PREPROCESS_MIN_RESERVATION_MB", 512),
    )


def _resident_floor() -> int:
    return _mb_env(
        "VLLM_GPU_PREPROCESS_RESIDENT_RESERVATION_MB",
        _env_int("VLLM_GPU_PREPROCESS_RESIDENT_MIN_MB", 64),
    )


def release_tokens(tokens: Iterable[Any]) -> int:
    return sum(
        1 for token in list(tokens)
        if hasattr(token, "release") and token.release()
    )


def reserve_preprocess(image: Any, label: str) -> _Reservation:
    token = _PREPROCESS_SEMAPHORE.acquire(max(_preprocess_floor(), _nbytes(image) * 4),
                                     label)
    release_decode_residency(image)
    return token


def reserve_resident(pixel_values: Any, label: str) -> _Reservation:
    return _RESIDENT_SEMAPHORE.acquire(max(_resident_floor(), _nbytes(pixel_values)),
                                  label)


def reserve_decode_batch(count: int, label: str) -> _Reservation:
    return _DECODE_SEMAPHORE.acquire(
        max(1, count) * _mb_env("VLLM_GPU_DECODE_RESERVATION_MB", 128),
        label,
    )


def decode_batch_limit(configured: int | None = None) -> int:
    limit = _env_int("VLLM_GPU_DECODE_MAX_BATCH_SIZE", 0) or 1
    return max(1, min(limit, int(configured))) if configured else max(1, limit)


def attach_decode_residency(tensor: Any, label: str) -> bool:
    token = _DECODE_RESIDENCY_SEMAPHORE.acquire(
        max(_mb_env("VLLM_GPU_DECODE_RESIDENCY_MB", 1), _nbytes(tensor)),
        label,
    )
    try:
        setattr(tensor, DECODE_TOKEN_ATTR, token)
        with contextlib.suppress(Exception):
            weakref.finalize(tensor, token.release)
        return True
    except Exception:
        token.release()
        return False


def release_decode_residency(obj: Any) -> int:
    token = getattr(obj, DECODE_TOKEN_ATTR, None)
    if token is None:
        return 0
    with contextlib.suppress(Exception):
        delattr(obj, DECODE_TOKEN_ATTR)
    return 1 if token.release() else 0


def _get(obj: Any, key: str) -> Any:
    try:
        return obj[key]
    except Exception:
        return getattr(obj, key, None)


def _set(obj: Any, key: str, value: Any) -> bool:
    try:
        obj[key] = value
        return True
    except Exception:
        with contextlib.suppress(Exception):
            setattr(obj, key, value)
            return True
    return False


def _attach(obj: Any, tokens: list[_Reservation]) -> bool:
    try:
        setattr(obj, TOKENS_ATTR, tokens)
        return True
    except Exception:
        return _set(obj, TOKENS_ATTR, tokens)


def _keep_gpu(mm_config: Any) -> bool:
    mode = os.environ.get("VLLM_GPU_PREPROCESS_PIXEL_VALUES_RESIDENCY", "").lower()
    if mode in {"gpu", "cuda"}:
        return True
    if mode in {"cpu", "host"}:
        return False
    return getattr(mm_config, "mm_tensor_ipc", None) == "torch_shm"


def finalize_hf_inputs(hf_inputs: Any, mm_config: Any,
                       tokens: Iterable[_Reservation]) -> str:
    scratch = list(tokens)
    pixel_values = _get(hf_inputs, "pixel_values")
    if _keep_gpu(mm_config):
        resident: list[_Reservation] = []
        try:
            if pixel_values is not None:
                resident.append(reserve_resident(pixel_values,
                                                 "qwen3_vl.pixel_values"))
            release_tokens(scratch)
            scratch = []
            if resident:
                if not _attach(hf_inputs, resident):
                    raise RuntimeError("failed to attach GPU-resident tokens")
                if pixel_values is not None:
                    _attach(pixel_values, resident)
        except BaseException:
            release_tokens(resident)
            release_tokens(scratch)
            raise
        return "gpu"
    if pixel_values is not None and str(getattr(pixel_values, "device", "")).startswith(
            "cuda"):
        _set(hf_inputs, "pixel_values", pixel_values.cpu())
    release_tokens(scratch)
    return "cpu"


def attach_tokens_to_items(hf_inputs: Any, items: Iterable[Any]) -> int:
    tokens = _pull_tokens(hf_inputs) or _pull_tokens(_get(hf_inputs, "pixel_values"))
    attached = 0
    for item, token in zip(list(items), tokens):
        try:
            setattr(item, TOKENS_ATTR, [token])
            attached += 1
        except Exception:
            token.release()
    release_tokens(tokens[attached:])
    return attached


def _pull_tokens(obj: Any, seen: set[int] | None = None) -> list[_Reservation]:
    if obj is None:
        return []
    if isinstance(obj, _Reservation):
        return [obj]
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    found: dict[int, _Reservation] = {}

    def add(values: Iterable[Any]) -> None:
        for value in values:
            if isinstance(value, _Reservation):
                found[id(value)] = value

    for attr in (TOKEN_ATTR, TOKENS_ATTR):
        tokens = getattr(obj, attr, None)
        if tokens is not None:
            add(tokens if isinstance(tokens, (list, tuple)) else [tokens])
            with contextlib.suppress(Exception):
                delattr(obj, attr)
    for feature in getattr(obj, "mm_features", None) or ():
        add(_pull_tokens(feature, seen))
        add(_pull_tokens(getattr(feature, "data", None), seen))
    values = obj if isinstance(obj, (list, tuple)) else None
    values_method = getattr(obj, "values", None)
    if values is None and callable(values_method):
        with contextlib.suppress(Exception):
            values = list(values_method())
    for value in values or ():
        add(_pull_tokens(value, seen))
    return list(found.values())


def _requests(obj: Any) -> Iterable[Any]:
    if obj is None:
        return ()
    return obj if isinstance(obj, (list, tuple)) else (obj,)


def track_requests(obj: Any) -> int:
    tracked = 0
    for request in _requests(obj):
        tokens = _pull_tokens(request)
        request_id = getattr(request, "request_id", None)
        if not tokens:
            continue
        if not isinstance(request_id, str) or not request_id:
            release_tokens(tokens)
            continue
        with _REQUEST_LOCK:
            if request_id in _FINISHED_EARLY:
                _FINISHED_EARLY.pop(request_id, None)
                release_now = True
            else:
                _REQUEST_TOKENS[request_id] = tokens
                tracked += 1
                release_now = False
        if release_now:
            release_tokens(tokens)
    return tracked


def release_request_tokens(obj: Any) -> int:
    return sum(release_tokens(_pull_tokens(request)) for request in _requests(obj))


def finish_outputs(outputs: Any) -> int:
    finished = {
        rid for rid in getattr(outputs, "finished_requests", None) or ()
        if isinstance(rid, str) and rid
    }
    for output in getattr(outputs, "outputs", None) or ():
        rid = getattr(output, "request_id", None)
        if getattr(output, "finished", False) and isinstance(rid, str) and rid:
            finished.add(rid)
    released: list[_Reservation] = []
    with _REQUEST_LOCK:
        for request_id in finished:
            tokens = _REQUEST_TOKENS.pop(request_id, None)
            if tokens is None:
                _FINISHED_EARLY[request_id] = None
                if len(_FINISHED_EARLY) > 4096:
                    _FINISHED_EARLY.pop(next(iter(_FINISHED_EARLY)), None)
            else:
                released.extend(tokens)
    return release_tokens(released)


def summary() -> dict[str, Any]:
    out = _PREPROCESS_SEMAPHORE.stats()
    for prefix, gate in (("resident", _RESIDENT_SEMAPHORE), ("decode", _DECODE_SEMAPHORE),
                         ("decode_residency", _DECODE_RESIDENCY_SEMAPHORE)):
        out.update({f"{prefix}_{key}": value for key, value in gate.stats().items()})
    with _REQUEST_LOCK:
        out["active_requests"] = len(_REQUEST_TOKENS)
        out["early_finished"] = len(_FINISHED_EARLY)
    return out


def _write_summary() -> None:
    path = os.environ.get("VLLM_GPU_PREPROCESS_SUMMARY_PATH",
                          "/workspace/results/weighted_admission_summary.json")
    data = summary()
    try:
        if os.path.dirname(path) and not os.path.isdir(os.path.dirname(path)):
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception:
        print("[weighted_admission_semaphore] failed to write summary", flush=True)
    # Keep Benchy's existing weighted-admission log parser working.
    for gate, prefix in (
            ("preprocess", ""),
            ("resident", "resident_"),
            ("decode", "decode_"),
            ("decode_residency", "decode_residency_")):
        print(
            "[weighted_admission] "
            f"gate={gate} "
            f"budget_bytes={data[prefix + 'budget_bytes']} "
            f"reserved_bytes={data[prefix + 'reserved_bytes']} "
            f"peak_reserved_bytes={data[prefix + 'peak_reserved_bytes']} "
            f"inflight={data[prefix + 'inflight']} "
            f"acquires={data[prefix + 'acquires']} "
            f"releases={data[prefix + 'releases']}",
            flush=True,
        )


atexit.register(_write_summary)
