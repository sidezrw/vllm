"""Weighted admission gate for GPU-resident image decode + preprocess.

The gate bounds APIServer-side GPU image/preprocess residency explicitly.
Reservations are held until the request's ZMQ send has completed, then released
from core_client.free_pending_messages().
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import weakref
from collections import Counter
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

TOKEN_ATTR = "_vllm_gpu_preprocess_reservation"
HF_RESERVATIONS_ATTR = "_vllm_gpu_preprocess_reservations"

_MIB = 1024 * 1024
_DEFAULT_BUDGET_MB = 8192
_DEFAULT_MIN_RESERVATION_MB = 512
_DEFAULT_SCRATCH_MB = 768
_DEFAULT_RESERVATION_SCALE = 1.0
_DEFAULT_DECODE_RESERVATION_MB = 512
_DEFAULT_DECODE_BUDGET_MB = 0
_DEFAULT_DECODE_RESIDENCY_MB = 64
_DEFAULT_DECODE_RESIDENCY_BUDGET_MB = 512
_DEFAULT_DECODE_BATCH_BUDGET_FRACTION = 0.50
_DEFAULT_DECODE_INFLIGHT_BUDGET_FRACTION = 0.33
_DEFAULT_DECODE_ENCODED_MULTIPLIER = 64
_DEFAULT_DECODE_RESULT_WAIT_TIMEOUT_S = 120.0
_DEFAULT_LOG_INTERVAL_S = 30.0
_DEFAULT_ORPHAN_TIMEOUT_MS = 10000
_DEFAULT_UNATTACHED_TIMEOUT_MS = 30000
_DEFAULT_DECODE_TENSOR_ORPHAN_TIMEOUT_MS = 30000
_DEFAULT_TORCH_MEMORY_CAP_MB = 0
_DEFAULT_TORCH_HEADROOM_MB = 512
_DEFAULT_API_MEMORY_CAP_MB = 0
_DEFAULT_DEVICE_HEADROOM_MB = 2048
_DEFAULT_CUDA_GUARD_POLL_MS = 50
_DEFAULT_CUDA_GUARD_CACHE_MS = 0
_DEFAULT_CUDA_GUARD_LOG_INTERVAL_MS = 5000
_DEFAULT_EMPTY_CACHE_MIN_INTERVAL_MS = 250
_TRACE_LIMIT = 8
_MEMTRACE_COUNTS: Counter[str] = Counter()
_MEMTRACE_LOCK = threading.Lock()
_TORCH_CAP_LOCK = threading.Lock()
_TORCH_CAP_STATE: dict[str, Any] = {}
_EMPTY_CACHE_LOCK = threading.Lock()
_EMPTY_CACHE_LAST_TS = 0.0
_CUDA_GUARD_COND = threading.Condition()
_CUDA_GUARD_WAITERS = 0
_CUDA_GUARD_PEAK_WAITERS = 0
_CUDA_GUARD_LAST_LOG_TS = 0.0
_CUDA_GUARD_SNAPSHOT_LOCK = threading.Lock()
_CUDA_GUARD_SNAPSHOT: dict[str, Any] = {
    "ts": 0.0,
    "snapshot": None,
}
_NVML_STATE_LOCK = threading.Lock()
_NVML_STATE: dict[str, Any] = {
    "init_attempted": False,
    "module": None,
    "error": "",
}
_PIXEL_RESIDENCY_LOG_LOCK = threading.Lock()
_PIXEL_RESIDENCY_LOGGED: set[str] = set()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("[weighted_admission] invalid %s=%r; using %s",
                       name, value, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("[weighted_admission] invalid %s=%r; using %s",
                       name, value, default)
        return default


def _trace_enabled() -> bool:
    return _env_bool("VLLM_GPU_PREPROCESS_TRACE", False)


def _memtrace_enabled() -> bool:
    return _env_bool("VLLM_GPU_PREPROCESS_MEMTRACE", False)


def _memtrace_every() -> int:
    return max(1, _env_int("VLLM_GPU_PREPROCESS_MEMTRACE_EVERY", 1))


def _memtrace_sync() -> bool:
    return _env_bool("VLLM_GPU_PREPROCESS_MEMTRACE_SYNC", False)


def _torch_memory_cap_mb() -> int:
    return max(
        0,
        _env_int(
            "VLLM_GPU_PREPROCESS_TORCH_MEMORY_CAP_MB",
            _DEFAULT_TORCH_MEMORY_CAP_MB,
        ),
    )


def _torch_headroom_mb() -> int:
    return max(
        0,
        _env_int(
            "VLLM_GPU_PREPROCESS_TORCH_HEADROOM_MB",
            _DEFAULT_TORCH_HEADROOM_MB,
        ),
    )


def _api_memory_cap_mb() -> int:
    configured = _env_int(
        "VLLM_GPU_PREPROCESS_API_MEMORY_CAP_MB",
        _DEFAULT_API_MEMORY_CAP_MB,
    )
    if configured > 0:
        return configured
    return _torch_memory_cap_mb()


def _device_headroom_mb() -> int:
    return max(
        0,
        _env_int(
            "VLLM_GPU_PREPROCESS_DEVICE_HEADROOM_MB",
            _DEFAULT_DEVICE_HEADROOM_MB,
        ),
    )


def _cuda_guard_poll_s() -> float:
    poll_ms = _env_int(
        "VLLM_GPU_PREPROCESS_CUDA_GUARD_POLL_MS",
        _DEFAULT_CUDA_GUARD_POLL_MS,
    )
    return max(0.001, poll_ms / 1000.0)


def _cuda_guard_cache_s() -> float:
    cache_ms = _env_int(
        "VLLM_GPU_PREPROCESS_CUDA_GUARD_CACHE_MS",
        _DEFAULT_CUDA_GUARD_CACHE_MS,
    )
    return max(0.0, cache_ms / 1000.0)


def _cuda_guard_log_interval_s() -> float:
    interval_ms = _env_int(
        "VLLM_GPU_PREPROCESS_CUDA_GUARD_LOG_INTERVAL_MS",
        _DEFAULT_CUDA_GUARD_LOG_INTERVAL_MS,
    )
    return max(0.1, interval_ms / 1000.0)


def _cuda_guard_disabled() -> bool:
    return _env_bool("VLLM_GPU_PREPROCESS_DISABLE_CUDA_GUARD", False)


def _preprocess_min_reservation_bytes() -> int:
    return max(
        1,
        _env_int(
            "VLLM_GPU_PREPROCESS_MIN_RESERVATION_MB",
            _DEFAULT_MIN_RESERVATION_MB,
        ),
    ) * _MIB


def _preprocess_scratch_bytes() -> int:
    return max(
        0,
        _env_int("VLLM_GPU_PREPROCESS_SCRATCH_MB", _DEFAULT_SCRATCH_MB),
    ) * _MIB


def _reservation_scale() -> float:
    return max(
        1.0,
        _env_float(
            "VLLM_GPU_PREPROCESS_RESERVATION_SCALE",
            _DEFAULT_RESERVATION_SCALE,
        ),
    )


def _empty_cache_on_idle() -> bool:
    return _env_bool("VLLM_GPU_PREPROCESS_EMPTY_CACHE_ON_IDLE", False)


def _empty_cache_on_pressure() -> bool:
    return _env_bool("VLLM_GPU_PREPROCESS_EMPTY_CACHE_ON_PRESSURE", True)


def _empty_cache_min_interval_s() -> float:
    interval_ms = _env_int(
        "VLLM_GPU_PREPROCESS_EMPTY_CACHE_MIN_INTERVAL_MS",
        _DEFAULT_EMPTY_CACHE_MIN_INTERVAL_MS,
    )
    return max(0.0, interval_ms / 1000.0)


def _direct_rpc_transport_mode() -> str:
    mode = os.environ.get(
        "VLLM_GPU_PREPROCESS_DIRECT_RPC_TRANSPORT_MODE",
        "explicit_cpu",
    )
    mode = str(mode).strip().lower().replace("-", "_")
    if mode in ("explicit", "explicit_cpu", "cpu", "cpu_spill"):
        return "explicit_cpu"
    if mode in ("pinned", "pinned_cpu", "pin_memory"):
        return "pinned_cpu"
    logger.warning(
        "[weighted_admission] invalid "
        "VLLM_GPU_PREPROCESS_DIRECT_RPC_TRANSPORT_MODE=%r; using explicit_cpu",
        mode,
    )
    return "explicit_cpu"


def _output_finish_release_enabled() -> bool:
    return _env_bool("VLLM_GPU_PREPROCESS_OUTPUT_FINISH_RELEASE", True)


def _pixel_values_residency_mode() -> str:
    mode = os.environ.get(
        "VLLM_GPU_PREPROCESS_PIXEL_VALUES_RESIDENCY",
        "auto",
    ).strip().lower()
    if mode in ("", "auto"):
        return "auto"
    if mode in ("cpu", "host", "direct_rpc"):
        return "cpu"
    if mode in ("gpu", "cuda", "torch_shm"):
        return "gpu"
    logger.warning(
        "[weighted_admission] invalid "
        "VLLM_GPU_PREPROCESS_PIXEL_VALUES_RESIDENCY=%r; using auto",
        mode,
    )
    return "auto"


def _mm_tensor_ipc_name(mm_config: Any) -> str:
    value = getattr(mm_config, "mm_tensor_ipc", None)
    if value is None:
        value = "direct_rpc"
    return str(value).strip().lower() or "direct_rpc"


def gpu_preprocess_keep_pixel_values_on_gpu(mm_config: Any) -> bool:
    mode = _pixel_values_residency_mode()
    if mode == "gpu":
        return True
    if mode == "cpu":
        return False
    return _mm_tensor_ipc_name(mm_config) == "torch_shm"


def _copy_cuda_tensor_to_pinned_cpu(value: Any) -> Any:
    try:
        import torch
    except Exception:
        return value.cpu()

    try:
        cpu_tensor = torch.empty_like(value, device="cpu", pin_memory=True)
        cpu_tensor.copy_(value, non_blocking=True)
        current_stream = getattr(torch.cuda, "current_stream", None)
        if callable(current_stream):
            try:
                stream = current_stream(getattr(value, "device", None))
            except TypeError:
                stream = current_stream()
            synchronize = getattr(stream, "synchronize", None)
            if callable(synchronize):
                synchronize()
            else:
                torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
        return cpu_tensor
    except Exception as exc:
        logger.warning(
            "[weighted_admission] pinned_cpu_spill_failed error=%s; "
            "falling back to tensor.cpu()",
            type(exc).__name__,
        )
        return value.cpu()


def _log_pixel_residency_once(mode: str, mm_tensor_ipc: str) -> None:
    key = f"{mode}:{mm_tensor_ipc}"
    with _PIXEL_RESIDENCY_LOG_LOCK:
        if key in _PIXEL_RESIDENCY_LOGGED:
            return
        _PIXEL_RESIDENCY_LOGGED.add(key)
    logger.info(
        "[weighted_admission] pixel_values_residency mode=%s "
        "mm_tensor_ipc=%s",
        mode,
        mm_tensor_ipc,
    )


def _trace_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return "none"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        rendered = [_trace_value(v) for v in values[:_TRACE_LIMIT]]
        if len(values) > _TRACE_LIMIT:
            rendered.append(f"...+{len(values) - _TRACE_LIMIT}")
        return ",".join(rendered)
    text = str(value)
    return text.replace(" ", "_")


def _trace(event: str, **fields: Any) -> None:
    if not _trace_enabled():
        return
    body = " ".join(
        f"{key}={_trace_value(value)}" for key, value in sorted(fields.items())
    )
    logger.warning("[weighted_admission_trace] event=%s %s", event, body)


def _rss_kb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return -1
    return -1


def _nvml_module() -> Any:
    with _NVML_STATE_LOCK:
        if _NVML_STATE["init_attempted"]:
            return _NVML_STATE["module"]
        _NVML_STATE["init_attempted"] = True

    module = None
    error = ""
    try:
        try:
            import pynvml as module  # type: ignore[no-redef]
        except Exception:
            from nvidia import ml as module  # type: ignore[no-redef]
        module.nvmlInit()
    except Exception as exc:
        module = None
        error = type(exc).__name__

    with _NVML_STATE_LOCK:
        _NVML_STATE["module"] = module
        _NVML_STATE["error"] = error
    if error:
        logger.warning("[weighted_admission] nvml_unavailable error=%s", error)
    return module


def _nvml_compute_processes(module: Any, handle: Any) -> Sequence[Any]:
    for name in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        getter = getattr(module, name, None)
        if getter is None:
            continue
        try:
            return getter(handle)
        except Exception:
            continue
    return ()


def _current_process_gpu_used_bytes(device: int) -> tuple[int, str]:
    module = _nvml_module()
    if module is None:
        with _NVML_STATE_LOCK:
            error = str(_NVML_STATE.get("error") or "unavailable")
        return -1, error

    try:
        handle = module.nvmlDeviceGetHandleByIndex(int(device))
        pid = os.getpid()
        for process in _nvml_compute_processes(module, handle):
            if getattr(process, "pid", None) != pid:
                continue
            used = getattr(process, "usedGpuMemory", None)
            if used is None:
                used = getattr(process, "usedGpuMemoryBytes", None)
            if used is None:
                return -1, "missing_usedGpuMemory"
            return int(used), "nvml_process"
        return 0, "nvml_process_absent"
    except Exception as exc:
        return -1, type(exc).__name__


def configure_torch_memory_cap(context: str = "") -> bool:
    """Apply an API-process PyTorch CUDA allocator cap when configured."""

    cap_mb = _torch_memory_cap_mb()
    if cap_mb <= 0:
        return False

    pid = os.getpid()
    with _TORCH_CAP_LOCK:
        if (
            _TORCH_CAP_STATE.get("pid") == pid
            and _TORCH_CAP_STATE.get("cap_mb") == cap_mb
        ):
            return True

    try:
        import torch
    except Exception:
        return False

    try:
        if not torch.cuda.is_available():
            return False
        device = torch.cuda.current_device()
        _free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        cap_bytes = cap_mb * _MIB
        fraction = max(0.01, min(1.0, cap_bytes / max(1, total_bytes)))
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    except Exception as exc:
        logger.warning(
            "[weighted_admission] torch_memory_cap_failed cap_mb=%s "
            "context=%s error=%s",
            cap_mb,
            context,
            type(exc).__name__,
        )
        return False

    with _TORCH_CAP_LOCK:
        _TORCH_CAP_STATE.update({
            "pid": pid,
            "cap_mb": cap_mb,
            "device": device,
            "fraction": fraction,
            "total_bytes": total_bytes,
        })
    logger.warning(
        "[weighted_admission] torch_memory_cap pid=%s device=%s cap_mb=%s "
        "fraction=%.6f total_bytes=%s context=%s",
        pid,
        device,
        cap_mb,
        fraction,
        total_bytes,
        context,
    )
    return True


def _admission_gate_stats() -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for name in ("_GATE", "_DECODE_GATE", "_DECODE_RESIDENCY_GATE"):
        gate = globals().get(name)
        if gate is None:
            continue
        try:
            stats.append(dict(gate.stats()))
        except Exception:
            continue
    return stats


def _all_admission_gates_idle() -> bool:
    return all(
        int(stats.get("reserved_bytes", 0)) == 0
        for stats in _admission_gate_stats()
    )


def _total_admission_reserved_bytes() -> int:
    return sum(
        int(stats.get("reserved_bytes", 0))
        for stats in _admission_gate_stats()
    )


def _notify_cuda_guard() -> None:
    with _CUDA_GUARD_COND:
        _CUDA_GUARD_COND.notify_all()


def maybe_empty_cuda_cache(context: str = "", *, require_idle: bool = True) -> bool:
    """Return cached PyTorch CUDA blocks on idle release or guard pressure."""

    global _EMPTY_CACHE_LAST_TS

    if require_idle:
        if not _empty_cache_on_idle():
            return False
    elif not (_empty_cache_on_pressure() or _empty_cache_on_idle()):
        return False
    if require_idle and not _all_admission_gates_idle():
        return False

    now = time.monotonic()
    with _EMPTY_CACHE_LOCK:
        if now - _EMPTY_CACHE_LAST_TS < _empty_cache_min_interval_s():
            return False
        _EMPTY_CACHE_LAST_TS = now

    try:
        import torch
    except Exception:
        return False

    try:
        if not torch.cuda.is_available():
            return False
        mem_trace("empty-cache-before", context=context, force=True)
        torch.cuda.empty_cache()
        mem_trace("empty-cache-after", context=context, force=True)
        _notify_cuda_guard()
        return True
    except Exception as exc:
        logger.warning(
            "[weighted_admission] empty_cache_failed context=%s error=%s",
            context,
            type(exc).__name__,
        )
        return False


def _torch_memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "torch_cuda_available": 0,
        "torch_device": -1,
        "torch_allocated_bytes": -1,
        "torch_reserved_bytes": -1,
        "torch_max_allocated_bytes": -1,
        "torch_max_reserved_bytes": -1,
        "cuda_free_bytes": -1,
        "cuda_total_bytes": -1,
        "cuda_used_bytes": -1,
        "cuda_used_minus_this_torch_reserved_bytes": -1,
        "process_gpu_used_bytes": -1,
        "process_gpu_used_source": "none",
        "process_gpu_used_minus_torch_reserved_bytes": -1,
    }
    try:
        import torch
    except Exception:
        return snapshot

    try:
        if not torch.cuda.is_available():
            return snapshot
        configure_torch_memory_cap("_torch_memory_snapshot")
        if _memtrace_sync():
            torch.cuda.synchronize()
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        used = total_bytes - free_bytes
        process_used, process_used_source = _current_process_gpu_used_bytes(
            device)
        snapshot.update({
            "torch_cuda_available": 1,
            "torch_device": device,
            "torch_allocated_bytes": torch.cuda.memory_allocated(device),
            "torch_reserved_bytes": reserved,
            "torch_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_max_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "cuda_free_bytes": free_bytes,
            "cuda_total_bytes": total_bytes,
            "cuda_used_bytes": used,
            "cuda_used_minus_this_torch_reserved_bytes": used - reserved,
            "process_gpu_used_bytes": process_used,
            "process_gpu_used_source": process_used_source,
            "process_gpu_used_minus_torch_reserved_bytes":
                process_used - reserved if process_used >= 0 else -1,
        })
    except Exception as exc:
        snapshot["torch_error"] = type(exc).__name__
    return snapshot


def _cuda_guard_memory_snapshot(*, force_live: bool = False) -> dict[str, Any]:
    """Return a live or cached memory snapshot for CUDA guard admission."""

    cache_s = _cuda_guard_cache_s()
    if cache_s <= 0:
        return _torch_memory_snapshot()

    now = time.monotonic()
    with _CUDA_GUARD_SNAPSHOT_LOCK:
        cached = _CUDA_GUARD_SNAPSHOT.get("snapshot")
        cached_ts = float(_CUDA_GUARD_SNAPSHOT.get("ts") or 0.0)
        if (
            not force_live
            and isinstance(cached, dict)
            and now - cached_ts < cache_s
        ):
            snapshot = dict(cached)
            snapshot["cuda_guard_cache_hit"] = 1
            snapshot["cuda_guard_cache_age_ms"] = int((now - cached_ts) * 1000)
            return snapshot

    snapshot = _torch_memory_snapshot()
    with _CUDA_GUARD_SNAPSHOT_LOCK:
        _CUDA_GUARD_SNAPSHOT["snapshot"] = dict(snapshot)
        _CUDA_GUARD_SNAPSHOT["ts"] = time.monotonic()
    snapshot["cuda_guard_cache_hit"] = 0
    snapshot["cuda_guard_cache_age_ms"] = 0
    return snapshot


def _api_used_for_cap(snapshot: dict[str, Any]) -> tuple[int, str]:
    process_used = int(snapshot.get("process_gpu_used_bytes", -1))
    if process_used >= 0:
        return process_used, str(snapshot.get("process_gpu_used_source", "nvml"))

    torch_reserved = int(snapshot.get("torch_reserved_bytes", -1))
    if torch_reserved >= 0:
        return torch_reserved, "torch_reserved_fallback"

    torch_allocated = int(snapshot.get("torch_allocated_bytes", -1))
    if torch_allocated >= 0:
        return torch_allocated, "torch_allocated_fallback"

    return -1, "unavailable"


def _log_cuda_guard_state(
    event: str,
    *,
    context: str,
    required_bytes: int,
    snapshot: dict[str, Any],
    api_used_bytes: int,
    api_used_source: str,
    api_cap_bytes: int,
    headroom_bytes: int,
    waited_s: float,
    force: bool = False,
) -> None:
    global _CUDA_GUARD_LAST_LOG_TS

    now = time.monotonic()
    if not force and now - _CUDA_GUARD_LAST_LOG_TS < _cuda_guard_log_interval_s():
        return
    _CUDA_GUARD_LAST_LOG_TS = now

    with _CUDA_GUARD_COND:
        waiters = _CUDA_GUARD_WAITERS
        peak_waiters = _CUDA_GUARD_PEAK_WAITERS

    logger.warning(
        "[weighted_admission_cuda_guard] event=%s context=%s "
        "required_bytes=%d api_cap_bytes=%d api_used_bytes=%d "
        "api_used_source=%s headroom_bytes=%d cuda_free_bytes=%s "
        "cuda_total_bytes=%s torch_allocated_bytes=%s "
        "torch_reserved_bytes=%s process_gpu_used_bytes=%s "
        "gate_total_reserved_bytes=%d waiters=%d peak_waiters=%d "
        "waited_s=%.3f",
        event,
        context,
        required_bytes,
        api_cap_bytes,
        api_used_bytes,
        api_used_source,
        headroom_bytes,
        snapshot.get("cuda_free_bytes", -1),
        snapshot.get("cuda_total_bytes", -1),
        snapshot.get("torch_allocated_bytes", -1),
        snapshot.get("torch_reserved_bytes", -1),
        snapshot.get("process_gpu_used_bytes", -1),
        _total_admission_reserved_bytes(),
        waiters,
        peak_waiters,
        waited_s,
    )


def wait_for_cuda_memory_headroom(
    required_bytes: int,
    *,
    context: str = "",
) -> bool:
    """Block admission until live API/device GPU memory can fit the request."""

    global _CUDA_GUARD_WAITERS, _CUDA_GUARD_PEAK_WAITERS

    if _cuda_guard_disabled():
        return True

    required = max(0, int(required_bytes))
    api_cap_bytes = _api_memory_cap_mb() * _MIB
    headroom_bytes = _device_headroom_mb() * _MIB
    if required <= 0 and api_cap_bytes <= 0 and headroom_bytes <= 0:
        return True

    configure_torch_memory_cap(f"cuda-guard:{context}")
    start = time.monotonic()
    logged_start = False

    with _CUDA_GUARD_COND:
        _CUDA_GUARD_WAITERS += 1
        _CUDA_GUARD_PEAK_WAITERS = max(
            _CUDA_GUARD_PEAK_WAITERS,
            _CUDA_GUARD_WAITERS,
        )

    try:
        while True:
            snapshot = _cuda_guard_memory_snapshot(force_live=logged_start)
            if int(snapshot.get("torch_cuda_available", 0)) != 1:
                return True

            api_used_bytes, api_used_source = _api_used_for_cap(snapshot)
            cuda_free_bytes = int(snapshot.get("cuda_free_bytes", -1))

            api_cap_ok = (
                api_cap_bytes <= 0
                or api_used_bytes < 0
                or api_used_bytes + required <= api_cap_bytes
            )
            free_ok = (
                cuda_free_bytes < 0
                or cuda_free_bytes >= required + headroom_bytes
            )

            waited_s = time.monotonic() - start
            if api_cap_ok and free_ok:
                if logged_start:
                    _log_cuda_guard_state(
                        "admit-after-wait",
                        context=context,
                        required_bytes=required,
                        snapshot=snapshot,
                        api_used_bytes=api_used_bytes,
                        api_used_source=api_used_source,
                        api_cap_bytes=api_cap_bytes,
                        headroom_bytes=headroom_bytes,
                        waited_s=waited_s,
                        force=True,
                    )
                return True

            maybe_empty_cuda_cache(
                f"cuda-guard-pressure:{context}",
                require_idle=False,
            )
            _log_cuda_guard_state(
                "wait",
                context=context,
                required_bytes=required,
                snapshot=snapshot,
                api_used_bytes=api_used_bytes,
                api_used_source=api_used_source,
                api_cap_bytes=api_cap_bytes,
                headroom_bytes=headroom_bytes,
                waited_s=waited_s,
                force=not logged_start,
            )
            logged_start = True
            with _CUDA_GUARD_COND:
                _CUDA_GUARD_COND.wait(timeout=_cuda_guard_poll_s())
    finally:
        with _CUDA_GUARD_COND:
            _CUDA_GUARD_WAITERS = max(0, _CUDA_GUARD_WAITERS - 1)


def describe_tensor(value: Any) -> str:
    if value is None:
        return "none"
    fields = [
        f"type={type(value).__name__}",
        f"id={hex(id(value))}",
    ]
    for attr in ("shape", "dtype", "device"):
        try:
            fields.append(f"{attr}={getattr(value, attr)}")
        except Exception:
            pass
    try:
        numel = value.numel()
        element_size = value.element_size()
        fields.append(f"numel={numel}")
        fields.append(f"element_size={element_size}")
        fields.append(f"nbytes={int(numel) * int(element_size)}")
    except Exception:
        try:
            nbytes = getattr(value, "nbytes")
            fields.append(f"nbytes={nbytes}")
        except Exception:
            pass
    return ";".join(str(field).replace(" ", "_") for field in fields)


def mem_trace(event: str, *, force: bool = False, **fields: Any) -> None:
    if not _memtrace_enabled():
        return
    with _MEMTRACE_LOCK:
        _MEMTRACE_COUNTS[event] += 1
        event_count = _MEMTRACE_COUNTS[event]
    every = _memtrace_every()
    if not force and every > 1 and event_count % every != 0:
        return

    try:
        import multiprocessing
        process_name = multiprocessing.current_process().name
    except Exception:
        process_name = "unknown"

    base_fields: dict[str, Any] = {
        "count": event_count,
        "pid": os.getpid(),
        "process": process_name,
        "thread": threading.current_thread().name,
        "rss_kb": _rss_kb(),
    }
    base_fields.update(_torch_memory_snapshot())

    gate = _GATE
    if gate is not None:
        try:
            stats = gate.stats()
            base_fields.update({
                "gate_reserved_bytes": stats["reserved_bytes"],
                "gate_peak_reserved_bytes": stats["peak_reserved_bytes"],
                "gate_inflight": stats["inflight"],
                "gate_acquires": stats["acquires"],
                "gate_releases": stats["releases"],
            })
        except Exception as exc:
            base_fields["gate_stats_error"] = type(exc).__name__

    decode_gate = globals().get("_DECODE_GATE")
    if decode_gate is not None:
        try:
            decode_stats = decode_gate.stats()
            base_fields.update({
                "decode_gate_reserved_bytes": decode_stats["reserved_bytes"],
                "decode_gate_peak_reserved_bytes":
                    decode_stats["peak_reserved_bytes"],
                "decode_gate_inflight": decode_stats["inflight"],
                "decode_gate_acquires": decode_stats["acquires"],
                "decode_gate_releases": decode_stats["releases"],
            })
        except Exception as exc:
            base_fields["decode_gate_stats_error"] = type(exc).__name__

    residency_gate = globals().get("_DECODE_RESIDENCY_GATE")
    if residency_gate is not None:
        try:
            residency_stats = residency_gate.stats()
            base_fields.update({
                "decode_residency_gate_reserved_bytes":
                    residency_stats["reserved_bytes"],
                "decode_residency_gate_peak_reserved_bytes":
                    residency_stats["peak_reserved_bytes"],
                "decode_residency_gate_inflight":
                    residency_stats["inflight"],
                "decode_residency_gate_acquires":
                    residency_stats["acquires"],
                "decode_residency_gate_releases":
                    residency_stats["releases"],
            })
        except Exception as exc:
            base_fields["decode_residency_gate_stats_error"] = (
                type(exc).__name__
            )

    base_fields.update(fields)
    body = " ".join(
        f"{key}={_trace_value(value)}"
        for key, value in sorted(base_fields.items())
    )
    logger.warning("[weighted_admission_mem] event=%s %s", event, body)


def _token_trace_id(token: Any) -> str:
    if token is None:
        return "none"
    return hex(id(token))


def _token_summary(tokens: Sequence[Any]) -> str:
    parts: list[str] = []
    for token in list(tokens)[:_TRACE_LIMIT]:
        if token is None:
            parts.append("none")
            continue
        released = getattr(token, "_released", None)
        parts.append(
            f"{hex(id(token))}:{getattr(token, 'nbytes', 'na')}:"
            f"{int(bool(released))}"
        )
    if len(tokens) > _TRACE_LIMIT:
        parts.append(f"...+{len(tokens) - _TRACE_LIMIT}")
    return ",".join(parts)


def _request_trace_id(request: Any) -> str:
    request_id = getattr(request, "request_id", None)
    if isinstance(request_id, str) and request_id:
        return request_id
    return f"object:{hex(id(request))}"


def _request_token_stats(request: Any) -> tuple[int, int, str]:
    features = list(getattr(request, "mm_features", None) or ())
    token_by_id: dict[int, Any] = {}
    for feature in features:
        feature_token = getattr(feature, TOKEN_ATTR, None)
        if feature_token is not None:
            token_by_id[id(feature_token)] = feature_token
        data = getattr(feature, "data", None)
        if data is not None:
            data_token = getattr(data, TOKEN_ATTR, None)
            if data_token is not None:
                token_by_id[id(data_token)] = data_token
    tokens = list(token_by_id.values())
    return len(features), len(tokens), _token_summary(tokens)


class ReservationToken:
    """Idempotent reservation handle returned by WeightedAdmissionGate."""

    def __init__(self, gate: "WeightedAdmissionGate | None", nbytes: int,
                 label: str = "", disabled: bool = False) -> None:
        self._gate = gate
        self.nbytes = int(nbytes)
        self.label = label
        self.disabled = disabled
        self._released = False
        self.release_on_send_completion = True
        self._lock = threading.Lock()

    def release(self) -> bool:
        try:
            unregister_unattached_reservations([self])
        except Exception:
            pass
        try:
            unregister_decode_tensor_reservations([self])
        except Exception:
            pass
        with self._lock:
            if self._released:
                return False
            self._released = True

        _release_decode_lane_for_token(self)
        if self._gate is not None and not self.disabled:
            self._gate._release(self)
        return True

    def resize(self, nbytes: int) -> bool:
        with self._lock:
            if self._released:
                return False

        if self._gate is not None and not self.disabled:
            self._gate._resize(self, int(nbytes))
        else:
            self.nbytes = int(nbytes)
        _release_decode_lane_for_token(self)
        return True

    def __enter__(self) -> "ReservationToken":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class WeightedAdmissionGate:
    """Thread-safe weighted semaphore measured in bytes."""

    def __init__(self, budget_bytes: int, *, disabled: bool = False,
                 min_reservation_bytes: int = 1, scratch_bytes: int = 0,
                 log_interval_s: float = _DEFAULT_LOG_INTERVAL_S,
                 name: str = "preprocess") -> None:
        self.name = str(name)
        self.budget_bytes = max(1, int(budget_bytes))
        self.min_reservation_bytes = max(1, int(min_reservation_bytes))
        self.scratch_bytes = max(0, int(scratch_bytes))
        self.disabled = bool(disabled)
        self.log_interval_s = float(log_interval_s)

        self._cond = threading.Condition()
        self._reserved_bytes = 0
        self._peak_reserved_bytes = 0
        self._inflight = 0
        self._acquire_count = 0
        self._release_count = 0
        self._last_log_ts = 0.0
        self._log_state("startup", force=True)

    def acquire(
        self,
        nbytes: int,
        label: str = "",
        *,
        apply_min: bool = True,
    ) -> ReservationToken:
        requested = int(nbytes)
        if apply_min:
            requested = max(self.min_reservation_bytes, requested)
        if self.disabled:
            self._log_state("disabled-acquire", force=False)
            return ReservationToken(None, requested, label=label, disabled=True)

        over_budget = requested > self.budget_bytes
        with self._cond:
            while True:
                if over_budget:
                    can_admit = self._inflight == 0
                else:
                    can_admit = (
                        self._reserved_bytes + requested <= self.budget_bytes
                    )
                if can_admit:
                    break
                self._cond.wait(timeout=min(5.0, self.log_interval_s))
                self._log_state("wait", force=False)

            self._reserved_bytes += requested
            self._peak_reserved_bytes = max(
                self._peak_reserved_bytes, self._reserved_bytes)
            self._inflight += 1
            self._acquire_count += 1
            token = ReservationToken(self, requested, label=label)

        try:
            wait_for_cuda_memory_headroom(
                requested,
                context=f"{self.name}.acquire:{label}",
            )
        except BaseException:
            token.release()
            raise

        self._log_state("acquire", force=False)
        mem_trace(
            "gate-acquire",
            label=label,
            requested=requested,
            reserved_bytes=self.stats()["reserved_bytes"],
            force=False,
        )
        _trace(
            "acquire",
            token=_token_trace_id(token),
            label=label,
            requested=requested,
            over_budget=over_budget,
            reserved_bytes=self.stats()["reserved_bytes"],
            inflight=self.stats()["inflight"],
        )
        return token

    def acquire_many(
        self,
        nbytes_by_token: Sequence[int],
        labels: Sequence[str] | None = None,
        *,
        apply_min: bool = True,
    ) -> list[ReservationToken]:
        requested = [
            max(self.min_reservation_bytes, int(nbytes))
            if apply_min
            else max(1, int(nbytes))
            for nbytes in nbytes_by_token
        ]
        if not requested:
            return []

        label_list = list(labels or ())
        if len(label_list) < len(requested):
            label_list.extend([""] * (len(requested) - len(label_list)))

        if self.disabled:
            self._log_state("disabled-acquire-many", force=False)
            return [
                ReservationToken(None, nbytes, label=label, disabled=True)
                for nbytes, label in zip(requested, label_list)
            ]

        total_requested = sum(requested)
        over_budget = total_requested > self.budget_bytes
        with self._cond:
            while True:
                if over_budget:
                    can_admit = self._inflight == 0
                else:
                    can_admit = (
                        self._reserved_bytes + total_requested
                        <= self.budget_bytes
                    )
                if can_admit:
                    break
                self._cond.wait(timeout=min(5.0, self.log_interval_s))
                self._log_state("wait-many", force=False)

            self._reserved_bytes += total_requested
            self._peak_reserved_bytes = max(
                self._peak_reserved_bytes, self._reserved_bytes)
            self._inflight += len(requested)
            self._acquire_count += len(requested)
            tokens = [
                ReservationToken(self, nbytes, label=label)
                for nbytes, label in zip(requested, label_list)
            ]

        try:
            wait_for_cuda_memory_headroom(
                total_requested,
                context=f"{self.name}.acquire_many:{len(tokens)}",
            )
        except BaseException:
            release_reservations(tokens)
            raise

        self._log_state("acquire-many", force=False)
        mem_trace(
            "gate-acquire-many",
            labels=",".join(label_list[:_TRACE_LIMIT]),
            requested=total_requested,
            tokens=len(tokens),
            reserved_bytes=self.stats()["reserved_bytes"],
            force=False,
        )
        _trace(
            "acquire-many",
            tokens=len(tokens),
            requested=total_requested,
            over_budget=over_budget,
            reserved_bytes=self.stats()["reserved_bytes"],
            inflight=self.stats()["inflight"],
        )
        return tokens

    def _resize(self, token: ReservationToken, nbytes: int) -> None:
        requested = max(self.min_reservation_bytes, int(nbytes))
        with self._cond:
            old_nbytes = token.nbytes
            while True:
                other_reserved = max(0, self._reserved_bytes - old_nbytes)
                if requested > self.budget_bytes:
                    can_resize = other_reserved == 0
                else:
                    can_resize = other_reserved + requested <= self.budget_bytes
                if can_resize:
                    break
                self._cond.wait(timeout=min(5.0, self.log_interval_s))
                self._log_state("wait-resize", force=False)

            self._reserved_bytes = other_reserved + requested
            self._peak_reserved_bytes = max(
                self._peak_reserved_bytes, self._reserved_bytes)
            token.nbytes = requested
            self._cond.notify_all()

        wait_for_cuda_memory_headroom(
            requested,
            context=f"{self.name}.resize:{token.label}",
        )
        self._log_state("resize", force=False)
        stats = self.stats()
        mem_trace(
            "gate-resize",
            label=token.label,
            old_nbytes=old_nbytes,
            nbytes=requested,
            reserved_bytes=stats["reserved_bytes"],
            force=False,
        )
        _trace(
            "resize",
            token=_token_trace_id(token),
            label=token.label,
            old_nbytes=old_nbytes,
            nbytes=requested,
            reserved_bytes=stats["reserved_bytes"],
            inflight=stats["inflight"],
        )

    def _release(self, token: ReservationToken) -> None:
        with self._cond:
            self._reserved_bytes -= token.nbytes
            if self._reserved_bytes < 0:
                logger.warning(
                    "[weighted_admission] reserved bytes underflow: %s",
                    self._reserved_bytes)
                self._reserved_bytes = 0
            self._inflight = max(0, self._inflight - 1)
            self._release_count += 1
            self._cond.notify_all()

        _notify_cuda_guard()
        maybe_empty_cuda_cache(
            f"{self.name}.release",
            require_idle=True,
        )
        self._log_state("release", force=False)
        stats = self.stats()
        mem_trace(
            "gate-release",
            label=token.label,
            nbytes=token.nbytes,
            reserved_bytes=stats["reserved_bytes"],
            releases=stats["releases"],
            force=False,
        )
        _trace(
            "release",
            token=_token_trace_id(token),
            label=token.label,
            nbytes=token.nbytes,
            reserved_bytes=stats["reserved_bytes"],
            inflight=stats["inflight"],
            releases=stats["releases"],
        )

    def _resize_many(
        self,
        tokens: Sequence[ReservationToken],
        nbytes_by_token: Sequence[int],
    ) -> None:
        if len(tokens) != len(nbytes_by_token):
            raise ValueError("tokens and nbytes_by_token lengths differ")
        if not tokens:
            return

        requested = [
            max(self.min_reservation_bytes, int(nbytes))
            for nbytes in nbytes_by_token
        ]
        total_requested = sum(requested)
        with self._cond:
            old_total = sum(int(token.nbytes) for token in tokens)
            while True:
                other_reserved = max(0, self._reserved_bytes - old_total)
                if total_requested > self.budget_bytes:
                    can_resize = other_reserved == 0
                else:
                    can_resize = (
                        other_reserved + total_requested <= self.budget_bytes
                    )
                if can_resize:
                    break
                self._cond.wait(timeout=min(5.0, self.log_interval_s))
                self._log_state("wait-resize-many", force=False)

            self._reserved_bytes = other_reserved + total_requested
            self._peak_reserved_bytes = max(
                self._peak_reserved_bytes, self._reserved_bytes)
            for token, nbytes in zip(tokens, requested):
                token.nbytes = nbytes
            self._cond.notify_all()

        wait_for_cuda_memory_headroom(
            total_requested,
            context=f"{self.name}.resize_many:{len(tokens)}",
        )
        self._log_state("resize-many", force=False)
        stats = self.stats()
        mem_trace(
            "gate-resize-many",
            tokens=len(tokens),
            old_nbytes=old_total,
            nbytes=total_requested,
            reserved_bytes=stats["reserved_bytes"],
            force=False,
        )
        _trace(
            "resize-many",
            tokens=len(tokens),
            old_nbytes=old_total,
            nbytes=total_requested,
            reserved_bytes=stats["reserved_bytes"],
            inflight=stats["inflight"],
        )

    def _log_state(self, event: str, *, force: bool) -> None:
        now = time.monotonic()
        if not force and (now - self._last_log_ts) < self.log_interval_s:
            return
        self._last_log_ts = now
        stats = self.stats()
        logger.warning(
            "[weighted_admission] gate=%s event=%s disabled=%s budget_bytes=%d "
            "min_reservation_bytes=%d scratch_bytes=%d reserved_bytes=%d "
            "peak_reserved_bytes=%d inflight=%d acquires=%d releases=%d",
            stats["name"],
            event,
            stats["disabled"],
            stats["budget_bytes"],
            stats["min_reservation_bytes"],
            stats["scratch_bytes"],
            stats["reserved_bytes"],
            stats["peak_reserved_bytes"],
            stats["inflight"],
            stats["acquires"],
            stats["releases"],
        )

    def stats(self) -> dict[str, int | bool | str]:
        with self._cond:
            return {
                "name": self.name,
                "budget_bytes": self.budget_bytes,
                "min_reservation_bytes": self.min_reservation_bytes,
                "scratch_bytes": self.scratch_bytes,
                "disabled": self.disabled,
                "reserved_bytes": self._reserved_bytes,
                "peak_reserved_bytes": self._peak_reserved_bytes,
                "inflight": self._inflight,
                "acquires": self._acquire_count,
                "releases": self._release_count,
            }


_GATE_LOCK = threading.Lock()
_GATE: WeightedAdmissionGate | None = None
_DECODE_GATE_LOCK = threading.Lock()
_DECODE_GATE: WeightedAdmissionGate | None = None
_DECODE_RESIDENCY_GATE_LOCK = threading.Lock()
_DECODE_RESIDENCY_GATE: WeightedAdmissionGate | None = None
_HF_RESERVATIONS_LOCK = threading.Lock()
_HF_RESERVATIONS_BY_ID: dict[int, tuple[ReservationToken, ...]] = {}
_HF_RESERVATIONS_BY_PIXEL_ID: dict[int, tuple[ReservationToken, ...]] = {}
_HF_RESERVATIONS_BY_PIXEL_STORAGE: dict[
    tuple[str, int], list[tuple[int, int, ReservationToken]]
] = {}
_HF_RESERVATION_TS_BY_ID: dict[int, float] = {}
_HF_RESERVATION_TS_BY_PIXEL_ID: dict[int, float] = {}
_HF_RESERVATION_TS_BY_PIXEL_STORAGE: dict[tuple[str, int], float] = {}
_HF_ORPHAN_REAPER_ACTIVE = False
_UNATTACHED_RESERVATIONS_LOCK = threading.Lock()
_UNATTACHED_RESERVATIONS_BY_TOKEN_ID: dict[int, ReservationToken] = {}
_UNATTACHED_RESERVATION_TS_BY_TOKEN_ID: dict[int, float] = {}
_UNATTACHED_REAPER_ACTIVE = False
_DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID: dict[int, ReservationToken] = {}
_DECODE_TENSOR_RESERVATION_TS_BY_TOKEN_ID: dict[int, float] = {}
_DECODE_TENSOR_IDS_BY_TOKEN_ID: dict[int, int] = {}
_DECODE_TENSOR_REAPER_ACTIVE = False
_TRACKED_SENDS_LOCK = threading.Lock()
_TRACKED_SENDS: list[tuple[Any, Any, bool]] = []
_TRACKED_SENDS_REAPER_ACTIVE = False
_REQUESTS_BY_ID_LOCK = threading.Lock()
_REQUESTS_BY_ID: dict[str, Any] = {}
_DECODE_RESERVATIONS_LOCK = threading.Lock()
_DECODE_RESERVATIONS_BY_TENSOR_ID: dict[int, Any] = {}


def _gate_disabled() -> bool:
    return os.environ.get("VLLM_GPU_PREPROCESS_DISABLE_GATE", "0") == "1"


def _configured_admission_budget_bytes() -> int:
    budget_mb = _env_int("VLLM_GPU_PREPROCESS_BUDGET_MB", _DEFAULT_BUDGET_MB)
    return max(1, budget_mb) * _MIB


def _raw_decode_residency_budget_bytes() -> int:
    budget_mb = _env_int(
        "VLLM_GPU_DECODE_RESIDENCY_BUDGET_MB",
        _DEFAULT_DECODE_RESIDENCY_BUDGET_MB,
    )
    return max(1, budget_mb) * _MIB


def _requested_decode_gate_budget_bytes(total_budget_bytes: int) -> int:
    explicit_mb = _env_int(
        "VLLM_GPU_DECODE_BUDGET_MB",
        _DEFAULT_DECODE_BUDGET_MB,
    )
    if explicit_mb > 0:
        configured_bytes = explicit_mb * _MIB
    else:
        configured_bytes = int(
            total_budget_bytes * _decode_inflight_budget_fraction()
        )
    floor_bytes = math.ceil(_decode_reservation_floor_bytes() *
                            _reservation_scale())
    return max(1, configured_bytes, floor_bytes)


def _admission_budget_partition() -> dict[str, int]:
    configured_budget = _configured_admission_budget_bytes()
    torch_cap_mb = _torch_memory_cap_mb()
    api_cap_mb = _api_memory_cap_mb()
    torch_headroom_mb = _torch_headroom_mb()
    total_budget = configured_budget
    effective_cap_mb = api_cap_mb if api_cap_mb > 0 else torch_cap_mb
    if effective_cap_mb > 0:
        cap_limited_budget = max(
            _preprocess_min_reservation_bytes(),
            (effective_cap_mb - torch_headroom_mb) * _MIB,
        )
        total_budget = min(total_budget, cap_limited_budget)

    min_preprocess = _preprocess_min_reservation_bytes()
    requested_decode = _requested_decode_gate_budget_bytes(total_budget)
    requested_residency = _raw_decode_residency_budget_bytes()

    decode_budget = min(
        requested_decode,
        max(1, total_budget - min_preprocess),
    )
    residency_budget = min(
        requested_residency,
        max(1, total_budget - min_preprocess - decode_budget),
    )
    preprocess_budget = max(
        min_preprocess,
        total_budget - decode_budget - residency_budget,
    )
    return {
        "configured_budget_bytes": configured_budget,
        "total_budget_bytes": total_budget,
        "preprocess_budget_bytes": preprocess_budget,
        "decode_budget_bytes": decode_budget,
        "decode_residency_budget_bytes": residency_budget,
        "requested_decode_budget_bytes": requested_decode,
        "requested_decode_residency_budget_bytes": requested_residency,
        "torch_cap_mb": torch_cap_mb,
        "api_cap_mb": api_cap_mb,
        "torch_headroom_mb": torch_headroom_mb,
    }


def get_gate() -> WeightedAdmissionGate:
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            partition = _admission_budget_partition()
            budget_bytes = partition["preprocess_budget_bytes"]
            disabled = _gate_disabled()
            min_bytes = _preprocess_min_reservation_bytes()
            scratch_bytes = _preprocess_scratch_bytes()
            if partition["total_budget_bytes"] < partition["configured_budget_bytes"]:
                logger.warning(
                    "[weighted_admission] effective_budget_clamped "
                    "configured_budget_mb=%s effective_total_budget_mb=%s "
                    "torch_cap_mb=%s api_cap_mb=%s torch_headroom_mb=%s",
                    math.ceil(partition["configured_budget_bytes"] / _MIB),
                    math.ceil(partition["total_budget_bytes"] / _MIB),
                    partition["torch_cap_mb"],
                    partition["api_cap_mb"],
                    partition["torch_headroom_mb"],
                )
            logger.warning(
                "[weighted_admission] admission_budget_partition "
                "configured_budget_mb=%s total_budget_mb=%s "
                "preprocess_budget_mb=%s decode_budget_mb=%s "
                "decode_residency_budget_mb=%s "
                "requested_decode_budget_mb=%s "
                "requested_decode_residency_budget_mb=%s "
                "api_cap_mb=%s device_headroom_mb=%s",
                math.ceil(partition["configured_budget_bytes"] / _MIB),
                math.ceil(partition["total_budget_bytes"] / _MIB),
                math.ceil(partition["preprocess_budget_bytes"] / _MIB),
                math.ceil(partition["decode_budget_bytes"] / _MIB),
                math.ceil(partition["decode_residency_budget_bytes"] / _MIB),
                math.ceil(partition["requested_decode_budget_bytes"] / _MIB),
                math.ceil(
                    partition["requested_decode_residency_budget_bytes"] /
                    _MIB
                ),
                partition["api_cap_mb"],
                _device_headroom_mb(),
            )
            _GATE = WeightedAdmissionGate(
                budget_bytes,
                disabled=disabled,
                min_reservation_bytes=min_bytes,
                scratch_bytes=scratch_bytes,
                name="preprocess",
            )
        return _GATE


def _decode_inflight_budget_fraction() -> float:
    return max(
        0.05,
        min(
            1.0,
            _env_float(
                "VLLM_GPU_DECODE_INFLIGHT_BUDGET_FRACTION",
                _DEFAULT_DECODE_INFLIGHT_BUDGET_FRACTION,
            ),
        ),
    )


def _decode_result_wait_timeout_s() -> float:
    return max(
        0.0,
        _env_float(
            "VLLM_GPU_DECODE_RESULT_WAIT_TIMEOUT_S",
            _DEFAULT_DECODE_RESULT_WAIT_TIMEOUT_S,
        ),
    )


def wait_for_decode_result(done_event: Any, context: str = "") -> bool:
    """Wait for a gated decode result without treating backpressure as failure.

    Upstream waits only five seconds after forcing a batch flush. That is too
    short once decode/preprocess is intentionally blocked by admission control.
    A zero timeout means wait indefinitely, otherwise wait with periodic logs.
    """

    timeout_s = _decode_result_wait_timeout_s()
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if timeout_s > 0 and elapsed >= timeout_s:
            logger.warning(
                "[weighted_admission] decode_result_wait_timeout "
                "context=%s elapsed_s=%.3f timeout_s=%.3f",
                context,
                elapsed,
                timeout_s,
            )
            mem_trace(
                "decode-result-wait-timeout",
                context=context,
                elapsed_s=f"{elapsed:.3f}",
                timeout_s=timeout_s,
                force=True,
            )
            return False

        if timeout_s > 0:
            wait_s = min(5.0, max(0.0, timeout_s - elapsed))
        else:
            wait_s = 5.0

        if done_event.wait(timeout=wait_s):
            mem_trace(
                "decode-result-wait-end",
                context=context,
                elapsed_s=f"{time.monotonic() - start:.3f}",
                timeout_s=timeout_s,
                force=False,
            )
            return True

        mem_trace(
            "decode-result-wait",
            context=context,
            elapsed_s=f"{time.monotonic() - start:.3f}",
            timeout_s=timeout_s,
            force=False,
        )


def get_decode_gate() -> WeightedAdmissionGate:
    """Limit decode-stage reservations across all renderer workers."""

    global _DECODE_GATE

    main_gate = get_gate()
    partition = _admission_budget_partition()
    budget_bytes = partition["decode_budget_bytes"]
    with _DECODE_GATE_LOCK:
        if (
            _DECODE_GATE is None
            or _DECODE_GATE.budget_bytes != budget_bytes
            or _DECODE_GATE.disabled != main_gate.disabled
        ):
            _DECODE_GATE = WeightedAdmissionGate(
                budget_bytes,
                disabled=main_gate.disabled,
                min_reservation_bytes=1,
                scratch_bytes=0,
                name="decode",
            )
        return _DECODE_GATE


def get_decode_residency_gate() -> WeightedAdmissionGate:
    """Limit decoded tensors waiting to be promoted into preprocess."""

    global _DECODE_RESIDENCY_GATE

    main_gate = get_gate()
    partition = _admission_budget_partition()
    budget_bytes = partition["decode_residency_budget_bytes"]
    with _DECODE_RESIDENCY_GATE_LOCK:
        if (
            _DECODE_RESIDENCY_GATE is None
            or _DECODE_RESIDENCY_GATE.budget_bytes != budget_bytes
            or _DECODE_RESIDENCY_GATE.disabled != main_gate.disabled
        ):
            _DECODE_RESIDENCY_GATE = WeightedAdmissionGate(
                budget_bytes,
                disabled=main_gate.disabled,
                min_reservation_bytes=1,
                scratch_bytes=0,
                name="decode_residency",
            )
        return _DECODE_RESIDENCY_GATE


def _release_decode_lane_for_token(token: Any) -> bool:
    lane_token = getattr(token, "_gpures_decode_lane_token", None)
    if lane_token is None:
        return False
    try:
        setattr(token, "_gpures_decode_lane_token", None)
    except Exception:
        pass
    try:
        released = bool(lane_token.release())
    except Exception:
        logger.exception("[weighted_admission] decode lane release failed")
        return False
    mem_trace(
        "decode-lane-release",
        token=_token_trace_id(token),
        released=released,
        force=False,
    )
    return released


def _attach_decode_lane_token(token: Any, lane_token: Any) -> None:
    if token is None or lane_token is None:
        return
    try:
        setattr(token, "_gpures_decode_lane_token", lane_token)
    except Exception:
        pass
    mem_trace(
        "decode-lane-attach",
        token=_token_trace_id(token),
        lane_token=_token_trace_id(lane_token),
        lane_nbytes=getattr(lane_token, "nbytes", -1),
        force=False,
    )


def reserve_bytes(nbytes: int, label: str = "") -> ReservationToken:
    configure_torch_memory_cap("reserve_bytes")
    return get_gate().acquire(nbytes, label=label)


def reserve_many(
    nbytes_by_token: Sequence[int],
    labels: Sequence[str] | None = None,
) -> list[ReservationToken]:
    configure_torch_memory_cap("reserve_many")
    return get_gate().acquire_many(nbytes_by_token, labels)


def resize_reservation(token: Any, nbytes: int, label: str = "") -> Any:
    if token is None or not hasattr(token, "resize"):
        return token
    if label:
        try:
            token.label = label
        except Exception:
            pass
    token.resize(nbytes)
    return token


def resize_reservations(
    tokens: Sequence[Any],
    nbytes_by_token: Sequence[int],
    labels: Sequence[str] | None = None,
) -> list[Any]:
    token_list = list(tokens)
    if len(token_list) != len(nbytes_by_token):
        raise ValueError("tokens and nbytes_by_token lengths differ")
    if not token_list:
        return token_list

    label_list = list(labels or ())
    if len(label_list) < len(token_list):
        label_list.extend([""] * (len(token_list) - len(label_list)))
    for token, label in zip(token_list, label_list):
        if token is not None and label:
            try:
                token.label = label
            except Exception:
                pass

    gate = getattr(token_list[0], "_gate", None)
    preprocess_gate = get_gate()
    if (
        gate is preprocess_gate
        and all(getattr(token, "_gate", None) is gate for token in token_list)
    ):
        gate._resize_many(token_list, nbytes_by_token)
    else:
        promoted_tokens = reserve_many(nbytes_by_token, label_list)
        release_reservations(token_list)
        _trace(
            "promote-reservations",
            old_tokens=len(token_list),
            new_tokens=len(promoted_tokens),
            old_gate=getattr(gate, "name", "none"),
            token_summary=_token_summary(tuple(promoted_tokens)),
        )
        token_list = promoted_tokens

    for token in token_list:
        _release_decode_lane_for_token(token)
    return token_list


def release_reservations(tokens: Iterable[Any]) -> int:
    tokens = list(tokens)
    unregister_unattached_reservations(tokens)
    released = 0
    for token in tokens:
        if token is not None and hasattr(token, "release"):
            try:
                _release_decode_lane_for_token(token)
                if token.release():
                    released += 1
            except Exception:
                logger.exception("[weighted_admission] token release failed")
    return released


def log_decode_result_shortfall(
    *,
    label: str,
    items_len: int,
    results_len: int,
    attached: int,
) -> None:
    """Emit an always-on diagnostic for native decode result mismatches."""

    logger.warning(
        "[weighted_admission] decode_result_shortfall label=%s items_len=%d "
        "results_len=%d attached=%d",
        label,
        int(items_len),
        int(results_len),
        int(attached),
    )


def log_decode_retry_exception(*, label: str, error: BaseException) -> None:
    """Emit an always-on diagnostic when singleton decode retry fails."""

    logger.warning(
        "[weighted_admission] decode_retry_exception label=%s error=%s",
        label,
        type(error).__name__,
    )


def _hf_orphan_timeout_s() -> float:
    timeout_ms = _env_int(
        "VLLM_GPU_PREPROCESS_ORPHAN_TIMEOUT_MS",
        _DEFAULT_ORPHAN_TIMEOUT_MS,
    )
    return max(0.0, timeout_ms / 1000.0)


def _unattached_timeout_s() -> float:
    timeout_ms = _env_int(
        "VLLM_GPU_PREPROCESS_UNATTACHED_TIMEOUT_MS",
        _DEFAULT_UNATTACHED_TIMEOUT_MS,
    )
    return max(0.0, timeout_ms / 1000.0)


def _decode_tensor_orphan_timeout_s() -> float:
    timeout_ms = _env_int(
        "VLLM_GPU_DECODE_TENSOR_ORPHAN_TIMEOUT_MS",
        _DEFAULT_DECODE_TENSOR_ORPHAN_TIMEOUT_MS,
    )
    return max(0.0, timeout_ms / 1000.0)


def _drop_tracked_request(request: Any) -> None:
    with _TRACKED_SENDS_LOCK:
        if not _TRACKED_SENDS:
            return
        before = len(_TRACKED_SENDS)
        _TRACKED_SENDS[:] = [
            entry for entry in _TRACKED_SENDS if entry[1] is not request
        ]
        after = len(_TRACKED_SENDS)
    if before != after:
        _trace(
            "drop-tracked",
            request_id=_request_trace_id(request),
            before=before,
            after=after,
        )


def _remove_hf_entries_for_tokens_locked(tokens: Sequence[Any]) -> int:
    token_ids = {id(token) for token in tokens if token is not None}
    if not token_ids:
        return 0

    removed = 0
    for mapping, timestamps in (
        (_HF_RESERVATIONS_BY_ID, _HF_RESERVATION_TS_BY_ID),
        (_HF_RESERVATIONS_BY_PIXEL_ID, _HF_RESERVATION_TS_BY_PIXEL_ID),
    ):
        stale_keys = [
            key for key, mapped_tokens in mapping.items()
            if any(id(token) in token_ids for token in mapped_tokens)
        ]
        for key in stale_keys:
            mapping.pop(key, None)
            timestamps.pop(key, None)
            removed += 1
    stale_storage_keys = [
        key for key, ranges in _HF_RESERVATIONS_BY_PIXEL_STORAGE.items()
        if any(id(token) in token_ids for _start, _stop, token in ranges)
    ]
    for key in stale_storage_keys:
        _HF_RESERVATIONS_BY_PIXEL_STORAGE.pop(key, None)
        _HF_RESERVATION_TS_BY_PIXEL_STORAGE.pop(key, None)
        removed += 1
    return removed


def _tensor_storage_key(tensor: Any) -> tuple[str, int] | None:
    """Return a stable-ish key for tensor views sharing one allocation."""

    storage = None
    try:
        storage = tensor.untyped_storage()
    except Exception:
        try:
            storage = tensor.storage()
        except Exception:
            storage = None
    if storage is None:
        return None

    data_ptr = getattr(storage, "data_ptr", None)
    if not callable(data_ptr):
        return None
    try:
        ptr = int(data_ptr())
    except Exception:
        return None
    if ptr <= 0:
        return None

    return (str(getattr(tensor, "device", "")), ptr)


def _tensor_storage_range(tensor: Any) -> tuple[tuple[str, int], int, int] | None:
    key = _tensor_storage_key(tensor)
    if key is None:
        return None
    try:
        start = int(tensor.storage_offset())
    except Exception:
        start = 0
    try:
        numel = int(tensor.numel())
    except Exception:
        shape = getattr(tensor, "shape", ())
        try:
            numel = _numel(tuple(int(dim) for dim in shape))
        except Exception:
            return None
    return key, start, start + max(0, numel)


def _grid_patch_counts(grid_thw: Any, count: int) -> list[int]:
    if count <= 0:
        return []

    rows = grid_thw
    tolist = getattr(rows, "tolist", None)
    if callable(tolist):
        try:
            rows = tolist()
        except Exception:
            rows = grid_thw

    patch_counts: list[int] = []
    try:
        iterable = list(rows)
    except Exception:
        return []
    for row in iterable[:count]:
        row_values = row
        row_tolist = getattr(row_values, "tolist", None)
        if callable(row_tolist):
            try:
                row_values = row_tolist()
            except Exception:
                pass
        try:
            values = list(row_values)
        except Exception:
            return []
        product = 1
        for value in values:
            try:
                product *= int(value)
            except Exception:
                return []
        patch_counts.append(max(1, product))
    return patch_counts


def _register_pixel_storage_reservations_locked(
    pixel_values: Any,
    grid_thw: Any,
    tokens: Sequence[ReservationToken],
    now: float,
) -> int:
    tensor_range = _tensor_storage_range(pixel_values)
    if tensor_range is None:
        return 0
    key, base_offset, _stop = tensor_range

    patch_counts = _grid_patch_counts(grid_thw, len(tokens))
    if len(patch_counts) < len(tokens):
        return 0

    try:
        stride0 = int(pixel_values.stride(0))
    except Exception:
        shape = tuple(int(dim) for dim in getattr(pixel_values, "shape", ()))
        stride0 = _numel(shape[1:]) if len(shape) > 1 else 1
    stride0 = max(1, stride0)

    ranges = _HF_RESERVATIONS_BY_PIXEL_STORAGE.setdefault(key, [])
    cursor = int(base_offset)
    for patches, token in zip(patch_counts, tokens):
        next_cursor = cursor + int(patches) * stride0
        ranges.append((cursor, next_cursor, token))
        cursor = next_cursor
    _HF_RESERVATION_TS_BY_PIXEL_STORAGE[key] = now
    return len(tokens)


def _pop_tensor_storage_reservation(tensor: Any) -> Any:
    tensor_range = _tensor_storage_range(tensor)
    if tensor_range is None:
        return None
    key, start, stop = tensor_range

    with _HF_RESERVATIONS_LOCK:
        ranges = _HF_RESERVATIONS_BY_PIXEL_STORAGE.get(key)
        if not ranges:
            return None

        found: Any = None
        remaining: list[tuple[int, int, ReservationToken]] = []
        for range_start, range_stop, token in ranges:
            overlaps = start < range_stop and stop > range_start
            if found is None and overlaps:
                found = token
            else:
                remaining.append((range_start, range_stop, token))

        if remaining:
            _HF_RESERVATIONS_BY_PIXEL_STORAGE[key] = remaining
        else:
            _HF_RESERVATIONS_BY_PIXEL_STORAGE.pop(key, None)
            _HF_RESERVATION_TS_BY_PIXEL_STORAGE.pop(key, None)

    if found is not None:
        _trace(
            "pop-pixel-storage",
            tensor_id=hex(id(tensor)),
            storage_key=f"{key[0]}:{key[1]}",
            start=start,
            stop=stop,
            token=_token_trace_id(found),
        )
    return found


def _start_hf_orphan_reaper_locked() -> None:
    global _HF_ORPHAN_REAPER_ACTIVE

    if _HF_ORPHAN_REAPER_ACTIVE or _hf_orphan_timeout_s() <= 0:
        return
    _HF_ORPHAN_REAPER_ACTIVE = True
    thread = threading.Thread(
        target=_hf_orphan_reaper_loop,
        name="GPUPreprocessHFReservationReaper",
        daemon=True,
    )
    thread.start()


def _start_unattached_reaper_locked() -> None:
    global _UNATTACHED_REAPER_ACTIVE

    if _UNATTACHED_REAPER_ACTIVE or _unattached_timeout_s() <= 0:
        return
    _UNATTACHED_REAPER_ACTIVE = True
    thread = threading.Thread(
        target=_unattached_reaper_loop,
        name="GPUPreprocessUnattachedReservationReaper",
        daemon=True,
    )
    thread.start()


def register_unattached_reservation(token: Any) -> bool:
    """Track a token until it reaches the HF/input attachment path.

    This is a last-resort cancellation guard for reservations acquired in the
    Qwen image loop. Normal execution unregisters during HF attachment or
    explicit release; the reaper only fires when a token is stranded between
    acquisition and transfer.
    """

    if token is None or not hasattr(token, "release"):
        return False
    if getattr(token, "_released", False):
        return False

    now = time.monotonic()
    with _UNATTACHED_RESERVATIONS_LOCK:
        _UNATTACHED_RESERVATIONS_BY_TOKEN_ID[id(token)] = token
        _UNATTACHED_RESERVATION_TS_BY_TOKEN_ID[id(token)] = now
        _start_unattached_reaper_locked()
        map_size = len(_UNATTACHED_RESERVATIONS_BY_TOKEN_ID)

    _trace(
        "register-unattached",
        token=_token_trace_id(token),
        nbytes=getattr(token, "nbytes", "na"),
        map_size=map_size,
    )
    return True


def unregister_unattached_reservations(tokens: Iterable[Any]) -> int:
    token_ids = [id(token) for token in tokens if token is not None]
    if not token_ids:
        return 0

    removed = 0
    with _UNATTACHED_RESERVATIONS_LOCK:
        for token_id in token_ids:
            if _UNATTACHED_RESERVATIONS_BY_TOKEN_ID.pop(token_id, None) is not None:
                removed += 1
            _UNATTACHED_RESERVATION_TS_BY_TOKEN_ID.pop(token_id, None)
        map_size = len(_UNATTACHED_RESERVATIONS_BY_TOKEN_ID)

    if removed:
        _trace(
            "unregister-unattached",
            removed=removed,
            map_size=map_size,
        )
    return removed


def _unattached_reaper_loop() -> None:
    global _UNATTACHED_REAPER_ACTIVE

    while True:
        timeout_s = _unattached_timeout_s()
        if timeout_s <= 0:
            with _UNATTACHED_RESERVATIONS_LOCK:
                _UNATTACHED_REAPER_ACTIVE = False
            return

        now = time.monotonic()
        stale_tokens: list[ReservationToken] = []
        with _UNATTACHED_RESERVATIONS_LOCK:
            if not _UNATTACHED_RESERVATIONS_BY_TOKEN_ID:
                _UNATTACHED_REAPER_ACTIVE = False
                return

            stale_token_ids = [
                token_id for token_id, ts in
                list(_UNATTACHED_RESERVATION_TS_BY_TOKEN_ID.items())
                if now - ts >= timeout_s
            ]
            for token_id in stale_token_ids:
                token = _UNATTACHED_RESERVATIONS_BY_TOKEN_ID.pop(
                    token_id, None)
                _UNATTACHED_RESERVATION_TS_BY_TOKEN_ID.pop(token_id, None)
                if token is not None:
                    stale_tokens.append(token)
            map_size = len(_UNATTACHED_RESERVATIONS_BY_TOKEN_ID)

        if stale_tokens:
            released = release_reservations(stale_tokens)
            logger.warning(
                "[weighted_admission] unattached_reservation_reaper "
                "released=%d stale=%d map_size=%d timeout_s=%.3f",
                released,
                len(stale_tokens),
                map_size,
                timeout_s,
            )
            _trace(
                "orphan-unattached-release",
                tokens=len(stale_tokens),
                released=released,
                map_size=map_size,
                token_summary=_token_summary(stale_tokens),
            )

        time.sleep(min(timeout_s, 1.0))


def _start_decode_tensor_reaper_locked() -> None:
    global _DECODE_TENSOR_REAPER_ACTIVE

    if (_DECODE_TENSOR_REAPER_ACTIVE or
            _decode_tensor_orphan_timeout_s() <= 0):
        return
    _DECODE_TENSOR_REAPER_ACTIVE = True
    thread = threading.Thread(
        target=_decode_tensor_reaper_loop,
        name="GPUDecodeTensorReservationReaper",
        daemon=True,
    )
    thread.start()


def register_decode_tensor_reservation(tensor_id: int,
                                       token: Any) -> bool:
    if token is None or not hasattr(token, "release"):
        return False
    if getattr(token, "_released", False):
        return False
    if getattr(token, "_gpures_decode_transferred", False):
        return False

    now = time.monotonic()
    with _DECODE_RESERVATIONS_LOCK:
        _DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID[id(token)] = token
        _DECODE_TENSOR_RESERVATION_TS_BY_TOKEN_ID[id(token)] = now
        _DECODE_TENSOR_IDS_BY_TOKEN_ID[id(token)] = tensor_id
        _start_decode_tensor_reaper_locked()
        map_size = len(_DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID)

    _trace(
        "register-decode-tensor",
        tensor_id=hex(tensor_id),
        token=_token_trace_id(token),
        nbytes=getattr(token, "nbytes", "na"),
        map_size=map_size,
    )
    return True


def unregister_decode_tensor_reservations(tokens: Iterable[Any]) -> int:
    token_ids = [id(token) for token in tokens if token is not None]
    if not token_ids:
        return 0

    removed = 0
    with _DECODE_RESERVATIONS_LOCK:
        for token_id in token_ids:
            if _DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID.pop(
                    token_id, None) is not None:
                removed += 1
            _DECODE_TENSOR_RESERVATION_TS_BY_TOKEN_ID.pop(token_id, None)
            _DECODE_TENSOR_IDS_BY_TOKEN_ID.pop(token_id, None)
        map_size = len(_DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID)

    if removed:
        _trace(
            "unregister-decode-tensor",
            removed=removed,
            map_size=map_size,
        )
    return removed


def _decode_tensor_reaper_loop() -> None:
    global _DECODE_TENSOR_REAPER_ACTIVE

    while True:
        timeout_s = _decode_tensor_orphan_timeout_s()
        if timeout_s <= 0:
            with _DECODE_RESERVATIONS_LOCK:
                _DECODE_TENSOR_REAPER_ACTIVE = False
            return

        now = time.monotonic()
        stale: list[tuple[ReservationToken, int | None]] = []
        with _DECODE_RESERVATIONS_LOCK:
            if not _DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID:
                _DECODE_TENSOR_REAPER_ACTIVE = False
                return

            stale_token_ids = [
                token_id for token_id, ts in
                list(_DECODE_TENSOR_RESERVATION_TS_BY_TOKEN_ID.items())
                if now - ts >= timeout_s
            ]
            for token_id in stale_token_ids:
                token = _DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID.pop(
                    token_id, None)
                _DECODE_TENSOR_RESERVATION_TS_BY_TOKEN_ID.pop(
                    token_id, None)
                tensor_id = _DECODE_TENSOR_IDS_BY_TOKEN_ID.pop(
                    token_id, None)
                if token is not None:
                    stale.append((token, tensor_id))
            map_size = len(_DECODE_TENSOR_RESERVATIONS_BY_TOKEN_ID)

        stale_tokens = [
            token for token, _tensor_id in stale
            if not getattr(token, "_released", False)
            and not getattr(token, "_gpures_decode_transferred", False)
        ]
        if stale_tokens:
            released = release_reservations(stale_tokens)
            logger.warning(
                "[weighted_admission] decode_tensor_orphan_reaper "
                "released=%d stale=%d map_size=%d timeout_s=%.3f",
                released,
                len(stale_tokens),
                map_size,
                timeout_s,
            )
            _trace(
                "orphan-decode-tensor-release",
                tokens=len(stale_tokens),
                released=released,
                map_size=map_size,
                token_summary=_token_summary(stale_tokens),
                tensor_ids=",".join(
                    hex(tensor_id) for _token, tensor_id in stale
                    if tensor_id is not None
                ),
            )

        time.sleep(min(timeout_s, 1.0))


def _hf_orphan_reaper_loop() -> None:
    global _HF_ORPHAN_REAPER_ACTIVE

    while True:
        timeout_s = _hf_orphan_timeout_s()
        if timeout_s <= 0:
            with _HF_RESERVATIONS_LOCK:
                _HF_ORPHAN_REAPER_ACTIVE = False
            return

        now = time.monotonic()
        token_by_id: dict[int, Any] = {}
        stale_hf_ids: list[int] = []
        stale_pixel_ids: list[int] = []
        stale_storage_keys: list[tuple[str, int]] = []
        with _HF_RESERVATIONS_LOCK:
            if (
                not _HF_RESERVATIONS_BY_ID
                and not _HF_RESERVATIONS_BY_PIXEL_ID
                and not _HF_RESERVATIONS_BY_PIXEL_STORAGE
            ):
                _HF_ORPHAN_REAPER_ACTIVE = False
                return

            for hf_id, ts in list(_HF_RESERVATION_TS_BY_ID.items()):
                if now - ts >= timeout_s:
                    stale_hf_ids.append(hf_id)
                    for token in _HF_RESERVATIONS_BY_ID.get(hf_id, ()):
                        token_by_id[id(token)] = token

            for pixel_id, ts in list(_HF_RESERVATION_TS_BY_PIXEL_ID.items()):
                if now - ts >= timeout_s:
                    stale_pixel_ids.append(pixel_id)
                    for token in _HF_RESERVATIONS_BY_PIXEL_ID.get(pixel_id, ()):
                        token_by_id[id(token)] = token

            for storage_key, ts in list(
                    _HF_RESERVATION_TS_BY_PIXEL_STORAGE.items()):
                if now - ts >= timeout_s:
                    stale_storage_keys.append(storage_key)
                    for _start, _stop, token in (
                            _HF_RESERVATIONS_BY_PIXEL_STORAGE.get(
                                storage_key, ())):
                        token_by_id[id(token)] = token

            tokens = tuple(token_by_id.values())
            removed = _remove_hf_entries_for_tokens_locked(tokens)
            hf_map_size = len(_HF_RESERVATIONS_BY_ID)
            pixel_map_size = len(_HF_RESERVATIONS_BY_PIXEL_ID)
            storage_map_size = len(_HF_RESERVATIONS_BY_PIXEL_STORAGE)

        if tokens:
            released = release_reservations(tokens)
            _trace(
                "orphan-hf-release",
                tokens=len(tokens),
                released=released,
                removed_entries=removed,
                stale_hf_ids=len(stale_hf_ids),
                stale_pixel_ids=len(stale_pixel_ids),
                stale_storage_keys=len(stale_storage_keys),
                hf_map_size=hf_map_size,
                pixel_map_size=pixel_map_size,
                storage_map_size=storage_map_size,
                token_summary=_token_summary(tokens),
            )

        time.sleep(min(1.0, max(0.05, timeout_s / 4.0)))


def _shape_tuple(image: Any) -> tuple[int, ...]:
    shape = getattr(image, "shape", ())
    return tuple(int(dim) for dim in shape)


def _element_size(image: Any) -> int:
    element_size = getattr(image, "element_size", None)
    if callable(element_size):
        try:
            return int(element_size())
        except Exception:
            pass
    dtype = str(getattr(image, "dtype", ""))
    if "float64" in dtype or "int64" in dtype:
        return 8
    if "float32" in dtype or "int32" in dtype:
        return 4
    if "float16" in dtype or "bfloat16" in dtype or "int16" in dtype:
        return 2
    return 1


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _smart_resize(height: int, width: int, factor: int, min_pixels: int,
                  max_pixels: int) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "Absolute aspect ratio must be < 200, got "
            f"{max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _image_chw(image: Any) -> tuple[int, int, int]:
    shape = _shape_tuple(image)
    if len(shape) != 3:
        return 3, 64, 64

    if shape[2] in (1, 3, 4):
        if shape[0] not in (1, 3, 4) or shape[2] <= 4:
            return min(3, shape[2]), shape[0], shape[1]
    return min(3, shape[0]), shape[1], shape[2]


def estimate_image_reservation_bytes(image: Any, gpu_proc: Any) -> int:
    """Estimate decoded + GPU preprocess residency for one image."""

    shape = _shape_tuple(image)
    decoded_bytes = _numel(shape) * _element_size(image) if shape else 0

    channels, height, width = _image_chw(image)
    channels = 3 if channels == 1 else max(3, channels)
    factor = int(getattr(gpu_proc, "factor", 28))
    min_pixels = int(getattr(gpu_proc, "min_pixels", 56 * 56))
    max_pixels = int(getattr(gpu_proc, "max_pixels", 14 * 14 * 4 * 1280))
    patch_size = int(getattr(gpu_proc, "patch_size", 14))
    temporal_patch_size = int(getattr(gpu_proc, "temporal_patch_size", 2))

    resized_height, resized_width = _smart_resize(
        height, width, factor, min_pixels, max_pixels)
    workspace_bytes = channels * resized_height * resized_width * 4

    grid_t = 1
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patch_elems = (
        grid_t * grid_h * grid_w * channels * temporal_patch_size *
        patch_size * patch_size
    )
    patch_bytes = patch_elems * 4

    gate = get_gate()
    base_estimated = max(
        gate.min_reservation_bytes,
        decoded_bytes + workspace_bytes + patch_bytes + gate.scratch_bytes,
    )
    scale = _reservation_scale()
    estimated = math.ceil(base_estimated * scale)
    mem_trace(
        "estimate-image",
        image=describe_tensor(image),
        decoded_bytes=decoded_bytes,
        workspace_bytes=workspace_bytes,
        patch_bytes=patch_bytes,
        scratch_bytes=gate.scratch_bytes,
        base_estimated_bytes=base_estimated,
        reservation_scale=scale,
        estimated_bytes=estimated,
        resized_height=resized_height,
        resized_width=resized_width,
        force=False,
    )
    return estimated


def _decode_reservation_floor_bytes() -> int:
    mb = _env_int(
        "VLLM_GPU_DECODE_RESERVATION_MB",
        _DEFAULT_DECODE_RESERVATION_MB,
    )
    return max(1, mb) * _MIB


def _decode_residency_floor_bytes() -> int:
    mb = _env_int(
        "VLLM_GPU_DECODE_RESIDENCY_MB",
        _DEFAULT_DECODE_RESIDENCY_MB,
    )
    return max(1, mb) * _MIB


def _decode_encoded_multiplier() -> int:
    return max(
        1,
        _env_int(
            "VLLM_GPU_DECODE_ENCODED_MULTIPLIER",
            _DEFAULT_DECODE_ENCODED_MULTIPLIER,
        ),
    )


def _decode_batch_budget_fraction() -> float:
    return max(
        0.05,
        min(
            1.0,
            _env_float(
                "VLLM_GPU_DECODE_BATCH_BUDGET_FRACTION",
                _DEFAULT_DECODE_BATCH_BUDGET_FRACTION,
            ),
        ),
    )


def _payload_nbytes(payload: Any) -> int:
    for attr in ("nbytes", "__len__"):
        try:
            value = getattr(payload, attr)
        except Exception:
            continue
        try:
            if callable(value):
                return int(value())
            return int(value)
        except Exception:
            continue
    return 0


def estimate_decode_reservation_bytes(payload: Any) -> int:
    """Estimate native decode scratch while a batch is actively decoding."""

    encoded_bytes = max(0, _payload_nbytes(payload))
    base_estimated = max(
        _decode_reservation_floor_bytes(),
        encoded_bytes * _decode_encoded_multiplier(),
    )
    scale = _reservation_scale()
    estimated = math.ceil(base_estimated * scale)
    mem_trace(
        "estimate-decode",
        encoded_bytes=encoded_bytes,
        base_estimated_bytes=base_estimated,
        reservation_scale=scale,
        estimated_bytes=estimated,
        force=False,
    )
    return estimated


def estimate_decode_residency_bytes(payload: Any) -> int:
    """Estimate decoded tensor residency before Qwen preprocess takes over."""

    encoded_bytes = max(0, _payload_nbytes(payload))
    base_estimated = max(
        _decode_residency_floor_bytes(),
        encoded_bytes * _decode_encoded_multiplier(),
    )
    scale = _reservation_scale()
    estimated = math.ceil(base_estimated * scale)
    mem_trace(
        "estimate-decode-residency",
        encoded_bytes=encoded_bytes,
        base_estimated_bytes=base_estimated,
        reservation_scale=scale,
        estimated_bytes=estimated,
        force=False,
    )
    return estimated


def plan_decode_batch_slices(payloads: Sequence[Any]) -> list[tuple[int, int]]:
    """Split a native decode batch so decode holds only part of the budget.

    The reserved decode tensors must later grow into preprocess reservations.
    Leaving explicit headroom avoids every worker filling the whole gate with
    small decode tokens and then deadlocking while trying to resize them.
    """

    gate = get_gate()
    decode_gate = get_decode_gate()
    limit = max(
        gate.min_reservation_bytes,
        decode_gate.min_reservation_bytes,
        int(gate.budget_bytes * _decode_batch_budget_fraction()),
        1,
    )
    limit = min(limit, decode_gate.budget_bytes)
    estimates = [estimate_decode_reservation_bytes(item) for item in payloads]
    slices: list[tuple[int, int]] = []
    start = 0
    running = 0
    for index, estimate in enumerate(estimates):
        if index > start and running + estimate > limit:
            slices.append((start, index))
            start = index
            running = 0
        running += estimate
    if start < len(estimates):
        slices.append((start, len(estimates)))

    mem_trace(
        "decode-plan",
        payloads=len(payloads),
        slices=len(slices),
        limit_bytes=limit,
        decode_gate_budget_bytes=decode_gate.budget_bytes,
        decode_inflight_fraction=_decode_inflight_budget_fraction(),
        total_estimated_bytes=sum(estimates),
        fraction=_decode_batch_budget_fraction(),
        force=True,
    )
    return slices


def reserve_decode_batch(payloads: Sequence[Any],
                         label: str = "decode-batch") -> list[ReservationToken]:
    residency_estimates = [
        estimate_decode_residency_bytes(item) for item in payloads
    ]
    decode_estimates = [
        estimate_decode_reservation_bytes(item) for item in payloads
    ]
    labels = [f"{label}.{idx}" for idx in range(len(residency_estimates))]
    configure_torch_memory_cap("reserve_decode_batch")
    decode_labels = [f"decode-inflight.{item}" for item in labels]
    tokens = get_decode_residency_gate().acquire_many(
        residency_estimates,
        labels,
        apply_min=False,
    )
    try:
        decode_tokens = get_decode_gate().acquire_many(
            decode_estimates,
            decode_labels,
        )
    except BaseException:
        release_reservations(tokens)
        raise
    for token, decode_token in zip(tokens, decode_tokens):
        _attach_decode_lane_token(token, decode_token)
    mem_trace(
        "decode-reserve",
        payloads=len(payloads),
        tokens=len(tokens),
        residency_bytes=sum(token.nbytes for token in tokens),
        decode_lane_bytes=sum(token.nbytes for token in decode_tokens),
        force=True,
    )
    return tokens


def _release_decode_reservation_for_tensor_id(tensor_id: int,
                                              token: Any) -> None:
    with _DECODE_RESERVATIONS_LOCK:
        mapped = _DECODE_RESERVATIONS_BY_TENSOR_ID.pop(tensor_id, None)
    unregister_decode_tensor_reservations([token, mapped])
    tokens: list[Any] = []
    if mapped is not None and mapped is not token:
        tokens.append(mapped)
    if not getattr(token, "_gpures_decode_transferred", False):
        tokens.append(token)
    release_reservations(tokens)


def attach_decode_reservation_to_tensor(tensor: Any, token: Any) -> bool:
    if token is None:
        return False

    tensor_id = id(tensor)
    try:
        setattr(token, "_gpures_decode_transferred", False)
    except Exception:
        pass
    attached_attr = False
    try:
        setattr(tensor, TOKEN_ATTR, token)
        attached_attr = True
    except Exception:
        with _DECODE_RESERVATIONS_LOCK:
            _DECODE_RESERVATIONS_BY_TENSOR_ID[tensor_id] = token

    try:
        weakref.finalize(
            tensor,
            _release_decode_reservation_for_tensor_id,
            tensor_id,
            token,
        )
        finalizer = 1
    except Exception:
        finalizer = 0

    register_decode_tensor_reservation(tensor_id, token)
    _trace(
        "attach-decode-tensor",
        tensor_id=hex(tensor_id),
        token=_token_trace_id(token),
        nbytes=getattr(token, "nbytes", "na"),
        attr=attached_attr,
        finalizer=finalizer,
    )
    mem_trace(
        "attach-decode-tensor",
        tensor=describe_tensor(tensor),
        token_nbytes=getattr(token, "nbytes", -1),
        attr=attached_attr,
        finalizer=finalizer,
        force=False,
    )
    # The decoded-residency reservation now follows the tensor. Release the
    # decode-only lane here so a lost or delayed preprocess handoff cannot
    # block every future decode. Qwen promotion moves it into the preprocess
    # gate before GPU preprocess starts.
    _release_decode_lane_for_token(token)
    return True


def pop_decode_reservation_for_tensor(tensor: Any) -> Any:
    token = getattr(tensor, TOKEN_ATTR, None)
    if token is not None:
        try:
            setattr(tensor, TOKEN_ATTR, None)
        except Exception:
            pass
    with _DECODE_RESERVATIONS_LOCK:
        mapped = _DECODE_RESERVATIONS_BY_TENSOR_ID.pop(id(tensor), None)
    if token is None:
        token = mapped
    elif mapped is not None and mapped is not token:
        release_reservations([mapped])
    if token is not None:
        try:
            setattr(token, "_gpures_decode_transferred", True)
        except Exception:
            pass
        unregister_decode_tensor_reservations([token])

    _trace(
        "pop-decode-tensor",
        tensor_id=hex(id(tensor)),
        token=_token_trace_id(token),
        found=token is not None,
    )
    mem_trace(
        "pop-decode-tensor",
        tensor=describe_tensor(tensor),
        token_nbytes=getattr(token, "nbytes", -1) if token is not None else -1,
        found=token is not None,
        force=False,
    )
    return token


def attach_reservations_to_hf_inputs(hf_inputs: Any,
                                     tokens: Sequence[ReservationToken]) -> None:
    token_tuple = tuple(tokens)
    if not token_tuple:
        return
    unregister_unattached_reservations(token_tuple)

    pixel_values = _hf_get(hf_inputs, "pixel_values")
    image_grid_thw = _hf_get(hf_inputs, "image_grid_thw")
    now = time.monotonic()
    with _HF_RESERVATIONS_LOCK:
        _HF_RESERVATIONS_BY_ID[id(hf_inputs)] = token_tuple
        _HF_RESERVATION_TS_BY_ID[id(hf_inputs)] = now
        if pixel_values is not None:
            _HF_RESERVATIONS_BY_PIXEL_ID[id(pixel_values)] = token_tuple
            _HF_RESERVATION_TS_BY_PIXEL_ID[id(pixel_values)] = now
        storage_entries = _register_pixel_storage_reservations_locked(
            pixel_values,
            image_grid_thw,
            token_tuple,
            now,
        ) if pixel_values is not None else 0
        _start_hf_orphan_reaper_locked()
        by_id_size = len(_HF_RESERVATIONS_BY_ID)
        by_pixel_size = len(_HF_RESERVATIONS_BY_PIXEL_ID)
        by_storage_size = len(_HF_RESERVATIONS_BY_PIXEL_STORAGE)

    try:
        setattr(hf_inputs, HF_RESERVATIONS_ATTR, token_tuple)
    except Exception:
        pass
    _trace(
        "attach-hf",
        hf_id=hex(id(hf_inputs)),
        pixel_id=hex(id(pixel_values)) if pixel_values is not None else "none",
        tokens=len(token_tuple),
        token_summary=_token_summary(token_tuple),
        hf_map_size=by_id_size,
        pixel_map_size=by_pixel_size,
        storage_map_size=by_storage_size,
        storage_entries=storage_entries,
    )


def _hf_get(hf_inputs: Any, key: str) -> Any:
    getter = getattr(hf_inputs, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    try:
        return hf_inputs[key]
    except Exception:
        return None


def _hf_set(hf_inputs: Any, key: str, value: Any) -> bool:
    try:
        hf_inputs[key] = value
        return True
    except Exception:
        try:
            setattr(hf_inputs, key, value)
            return True
        except Exception:
            logger.exception(
                "[weighted_admission] failed to set hf_inputs[%s]", key)
            return False


def _is_cuda_tensor(value: Any) -> bool:
    is_cuda = getattr(value, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return is_cuda
    device = getattr(value, "device", None)
    return str(device).startswith("cuda")


def mark_reservations_output_finish_only(
    tokens: Sequence[ReservationToken],
) -> None:
    for token in tokens:
        try:
            token.release_on_send_completion = False
        except Exception:
            pass


def prepare_hf_inputs_for_transport(
    hf_inputs: Any,
    mm_config: Any,
    tokens: Sequence[ReservationToken],
) -> str:
    """Select CPU/direct_rpc or CUDA/torch_shm pixel_values residency.

    direct_rpc serializes tensors with ``tensor.cpu()`` internally. Spilling
    explicitly here drops API-side CUDA pixel_values before they can queue in
    vLLM waiting, then releases the preprocess reservation immediately.
    torch_shm keeps CUDA pixel_values and defers token release until request
    output finish for this first safe prototype.
    """

    token_tuple = tuple(tokens)
    mm_tensor_ipc = _mm_tensor_ipc_name(mm_config)
    keep_gpu = gpu_preprocess_keep_pixel_values_on_gpu(mm_config)
    mode = "torch_shm_cuda" if keep_gpu else "direct_rpc_cpu"
    _log_pixel_residency_once(mode, mm_tensor_ipc)

    pixel_values = _hf_get(hf_inputs, "pixel_values")
    before = describe_tensor(pixel_values)
    copied_to_cpu = False

    if keep_gpu:
        mark_reservations_output_finish_only(token_tuple)
        attach_reservations_to_hf_inputs(hf_inputs, token_tuple)
    else:
        direct_rpc_mode = _direct_rpc_transport_mode()
        if pixel_values is not None and _is_cuda_tensor(pixel_values):
            cpu_method = getattr(pixel_values, "cpu", None)
            if not callable(cpu_method):
                raise TypeError("pixel_values does not support .cpu()")
            mem_trace(
                "pixel-values-before-cpu-spill",
                pixel_values=before,
                force=False,
            )
            if direct_rpc_mode == "pinned_cpu":
                pixel_values = _copy_cuda_tensor_to_pinned_cpu(pixel_values)
                mode = "direct_rpc_pinned_cpu"
            else:
                pixel_values = cpu_method()
            if not _hf_set(hf_inputs, "pixel_values", pixel_values):
                raise RuntimeError("failed to store CPU pixel_values")
            copied_to_cpu = True
        release_reservations(token_tuple)
        maybe_empty_cuda_cache("pixel-values-cpu-spill", require_idle=True)

    after = describe_tensor(_hf_get(hf_inputs, "pixel_values"))
    _trace(
        "pixel-values-transport",
        mode=mode,
        mm_tensor_ipc=mm_tensor_ipc,
        keep_gpu=keep_gpu,
        copied_to_cpu=copied_to_cpu,
        before=before,
        after=after,
        tokens=len(token_tuple),
        token_summary=_token_summary(token_tuple),
    )
    mem_trace(
        "pixel-values-after-transport",
        mode=mode,
        mm_tensor_ipc=mm_tensor_ipc,
        before=before,
        after=after,
        copied_to_cpu=copied_to_cpu,
        tokens=len(token_tuple),
        force=False,
    )
    return mode


def _forget_hf_reservations(hf_inputs: Any) -> None:
    pixel_values = _hf_get(hf_inputs, "pixel_values")
    with _HF_RESERVATIONS_LOCK:
        _HF_RESERVATIONS_BY_ID.pop(id(hf_inputs), None)
        _HF_RESERVATION_TS_BY_ID.pop(id(hf_inputs), None)
        if pixel_values is not None:
            _HF_RESERVATIONS_BY_PIXEL_ID.pop(id(pixel_values), None)
            _HF_RESERVATION_TS_BY_PIXEL_ID.pop(id(pixel_values), None)


def _clear_token_attr(obj: Any) -> None:
    try:
        setattr(obj, TOKEN_ATTR, None)
    except Exception:
        pass


def _append_direct_token(tokens: list[Any], obj: Any) -> None:
    token = getattr(obj, TOKEN_ATTR, None)
    if token is not None:
        tokens.append(token)
        _clear_token_attr(obj)


def _append_data_reservation_tokens(tokens: list[Any], data: Any) -> None:
    if data is None:
        return

    _append_direct_token(tokens, data)
    storage_token = _pop_tensor_storage_reservation(data)
    if storage_token is not None:
        tokens.append(storage_token)

    values = None
    values_method = getattr(data, "values", None)
    if callable(values_method):
        try:
            values = list(values_method())
        except Exception:
            values = None
    elif isinstance(data, (list, tuple)):
        values = list(data)

    if not values:
        return

    for value in values:
        _append_direct_token(tokens, value)
        elem_data = getattr(value, "data", None)
        if elem_data is not None and elem_data is not data:
            _append_data_reservation_tokens(tokens, elem_data)


def attach_reservations_to_items(hf_inputs: Any, items: Sequence[Any]) -> None:
    tokens = getattr(hf_inputs, HF_RESERVATIONS_ATTR, None)
    source = "attr" if tokens else "none"
    if not tokens:
        with _HF_RESERVATIONS_LOCK:
            tokens = _HF_RESERVATIONS_BY_ID.pop(id(hf_inputs), None)
        source = "hf_id" if tokens else source
    if not tokens:
        pixel_values = _hf_get(hf_inputs, "pixel_values")
        if pixel_values is not None:
            with _HF_RESERVATIONS_LOCK:
                tokens = _HF_RESERVATIONS_BY_PIXEL_ID.pop(id(pixel_values), None)
            source = "pixel_id" if tokens else source
    if not tokens:
        _trace(
            "attach-items-miss",
            hf_id=hex(id(hf_inputs)),
            items=len(items),
        )
        return

    consumed = 0
    for item, token in zip(items, tokens):
        setattr(item, TOKEN_ATTR, token)
        consumed += 1

    if consumed < len(tokens):
        release_reservations(tokens[consumed:])

    try:
        delattr(hf_inputs, HF_RESERVATIONS_ATTR)
    except Exception:
        pass
    with _HF_RESERVATIONS_LOCK:
        removed_entries = _remove_hf_entries_for_tokens_locked(tuple(tokens))
    _forget_hf_reservations(hf_inputs)
    _trace(
        "attach-items",
        hf_id=hex(id(hf_inputs)),
        items=len(items),
        consumed=consumed,
        removed_entries=removed_entries,
        source=source,
        token_summary=_token_summary(tuple(tokens)),
    )


def attach_item_reservation_to_feature(feature: Any, item: Any) -> bool:
    token = getattr(item, TOKEN_ATTR, None)
    if token is None:
        _trace(
            "attach-feature-miss",
            feature_id=hex(id(feature)),
            item_id=hex(id(item)),
        )
        return False
    try:
        setattr(feature, TOKEN_ATTR, token)
    except Exception:
        _trace(
            "attach-feature-failed",
            feature_id=hex(id(feature)),
            item_id=hex(id(item)),
            token=_token_trace_id(token),
        )
        return False
    _trace(
        "attach-feature",
        feature_id=hex(id(feature)),
        item_id=hex(id(item)),
        token=_token_trace_id(token),
    )
    return True


def release_item_reservation(item: Any) -> int:
    token = getattr(item, TOKEN_ATTR, None)
    released = release_reservations([token])
    if token is not None:
        try:
            setattr(item, TOKEN_ATTR, None)
        except Exception:
            pass
    return released


def release_feature_reservation(feature: Any) -> int:
    data = getattr(feature, "data", None)
    tokens: list[Any] = []
    if data is not None:
        _append_data_reservation_tokens(tokens, data)
    _append_direct_token(tokens, feature)

    released = release_reservations(tokens)
    with _HF_RESERVATIONS_LOCK:
        removed_entries = _remove_hf_entries_for_tokens_locked(tuple(tokens))
    _trace(
        "release-feature",
        feature_id=hex(id(feature)),
        data_id=hex(id(data)) if data is not None else "none",
        tokens=_token_summary(tuple(tokens)),
        released=released,
        removed_entries=removed_entries,
    )
    mem_trace(
        "release-feature",
        feature_id=hex(id(feature)),
        data_id=hex(id(data)) if data is not None else "none",
        released=released,
        force=False,
    )
    return released


def _collect_request_direct_tokens(request: Any) -> list[Any]:
    token_by_id: dict[int, Any] = {}
    for feature in getattr(request, "mm_features", None) or ():
        feature_token = getattr(feature, TOKEN_ATTR, None)
        if feature_token is not None:
            token_by_id[id(feature_token)] = feature_token
        data = getattr(feature, "data", None)
        if data is not None:
            data_token = getattr(data, TOKEN_ATTR, None)
            if data_token is not None:
                token_by_id[id(data_token)] = data_token
    return list(token_by_id.values())


def _has_output_finish_only_reservation(request: Any) -> bool:
    for token in _collect_request_direct_tokens(request):
        released = bool(getattr(token, "_released", False))
        release_on_send = bool(
            getattr(token, "release_on_send_completion", True))
        if not released and not release_on_send:
            return True
    return False


def release_request_reservations(
    request: Any,
    *,
    clear_data: bool,
    release_deferred: bool = True,
) -> int:
    released = 0
    mm_features = getattr(request, "mm_features", None)
    if not mm_features:
        _trace(
            "release-request-skip",
            request_id=_request_trace_id(request),
            reason="no_mm_features",
            clear_data=clear_data,
        )
        _drop_tracked_request(request)
        return released

    if not release_deferred and _has_output_finish_only_reservation(request):
        track_request_for_output_completion(request)
        feature_count, token_count, token_summary = _request_token_stats(
            request)
        _trace(
            "release-request-deferred",
            request_id=_request_trace_id(request),
            features=feature_count,
            tokens=token_count,
            token_summary=token_summary,
            clear_data=clear_data,
        )
        mem_trace(
            "release-request-deferred",
            request_id=_request_trace_id(request),
            features=feature_count,
            tokens=token_count,
            force=False,
        )
        return 0

    feature_count, token_count, token_summary = _request_token_stats(request)
    _trace(
        "release-request-begin",
        request_id=_request_trace_id(request),
        features=feature_count,
        tokens=token_count,
        token_summary=token_summary,
        clear_data=clear_data,
    )
    mem_trace(
        "release-request-begin",
        request_id=_request_trace_id(request),
        features=feature_count,
        tokens=token_count,
        clear_data=clear_data,
        force=False,
    )

    for feature in mm_features:
        released += release_feature_reservation(feature)
        if clear_data:
            try:
                feature.data = None
            except Exception:
                logger.exception(
                    "[weighted_admission] failed to clear mm feature data")

    request_id = getattr(request, "request_id", None)
    if isinstance(request_id, str) and request_id:
        with _REQUESTS_BY_ID_LOCK:
            if _REQUESTS_BY_ID.get(request_id) is request:
                _REQUESTS_BY_ID.pop(request_id, None)
            request_map_size = len(_REQUESTS_BY_ID)
    else:
        with _REQUESTS_BY_ID_LOCK:
            request_map_size = len(_REQUESTS_BY_ID)
    _drop_tracked_request(request)
    _trace(
        "release-request-end",
        request_id=_request_trace_id(request),
        released=released,
        request_map_size=request_map_size,
    )
    mem_trace(
        "release-request-end",
        request_id=_request_trace_id(request),
        released=released,
        request_map_size=request_map_size,
        force=False,
    )
    maybe_empty_cuda_cache("release-request-end", require_idle=True)
    return released


def release_request_after_send(request: Any, *, clear_data: bool) -> int:
    """Release send-scoped reservations after API->engine handoff.

    CUDA tensor IPC reservations marked output-finish-only stay live until an
    EngineCoreOutputs finished signal arrives.
    """

    track_request_for_output_completion(request)
    return release_request_reservations(
        request,
        clear_data=clear_data,
        release_deferred=False,
    )


def track_request_for_output_completion(request: Any) -> bool:
    """Remember a request so EngineCoreOutputs.finished_requests can free it."""

    if not _output_finish_release_enabled():
        return False
    if not getattr(request, "mm_features", None):
        _trace(
            "track-request-skip",
            request_id=_request_trace_id(request),
            reason="no_mm_features",
        )
        return False
    request_id = getattr(request, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        _trace(
            "track-request-skip",
            request_id=_request_trace_id(request),
            reason="missing_request_id",
        )
        return False

    feature_count, token_count, token_summary = _request_token_stats(request)
    with _REQUESTS_BY_ID_LOCK:
        _REQUESTS_BY_ID[request_id] = request
        request_map_size = len(_REQUESTS_BY_ID)
    _trace(
        "track-request",
        request_id=request_id,
        features=feature_count,
        tokens=token_count,
        token_summary=token_summary,
        request_map_size=request_map_size,
    )
    return True


def release_finished_request_reservations(outputs: Any, *,
                                          clear_data: bool) -> int:
    """Release reservations for requests reported finished by EngineCore."""

    if not _output_finish_release_enabled():
        return 0

    with _REQUESTS_BY_ID_LOCK:
        if not _REQUESTS_BY_ID:
            return 0

    finished = set(getattr(outputs, "finished_requests", None) or ())
    output_count = 0
    output_finished_count = 0
    for output in getattr(outputs, "outputs", None) or ():
        output_count += 1
        if getattr(output, "finished", False):
            output_finished_count += 1
            request_id = getattr(output, "request_id", None)
            if isinstance(request_id, str) and request_id:
                finished.add(request_id)

    if not finished:
        return 0

    released = 0
    with _REQUESTS_BY_ID_LOCK:
        request_map_size_before = len(_REQUESTS_BY_ID)
        matched_ids = [
            request_id for request_id in finished
            if _REQUESTS_BY_ID.get(request_id) is not None
        ]
        unmatched_ids = [
            request_id for request_id in finished
            if _REQUESTS_BY_ID.get(request_id) is None
        ]
        requests = [
            _REQUESTS_BY_ID.pop(request_id, None)
            for request_id in list(finished)
        ]
        request_map_size_after = len(_REQUESTS_BY_ID)

    _trace(
        "finished-seen",
        finished=len(finished),
        matched=len(matched_ids),
        unmatched=len(unmatched_ids),
        output_count=output_count,
        output_finished_count=output_finished_count,
        matched_ids=matched_ids,
        unmatched_ids=unmatched_ids,
        request_map_size_before=request_map_size_before,
        request_map_size_after=request_map_size_after,
    )
    for request in requests:
        if request is not None:
            released += release_request_reservations(
                request, clear_data=clear_data)
    _trace(
        "finished-release",
        finished=len(finished),
        released=released,
    )
    return released


def release_request_when_tracker_done(tracker: Any, request: Any, *,
                                      clear_data: bool) -> bool:
    """Release a multimodal request as soon as pyzmq finishes with it.

    core_client.free_pending_messages() only runs on later engine sends. That is
    too late for an admission gate because the later request can block before it
    reaches the send path. A single lightweight reaper polls MessageTracker.done
    so the final in-flight batch is released without waiting for another send.
    """

    global _TRACKED_SENDS_REAPER_ACTIVE

    if not getattr(request, "mm_features", None):
        _trace(
            "tracker-skip",
            request_id=_request_trace_id(request),
            reason="no_mm_features",
        )
        return False

    track_request_for_output_completion(request)

    if getattr(tracker, "done", False):
        _trace(
            "tracker-immediate",
            request_id=_request_trace_id(request),
            tracker=hex(id(tracker)),
        )
        release_request_after_send(request, clear_data=clear_data)
        return True

    with _TRACKED_SENDS_LOCK:
        _TRACKED_SENDS.append((tracker, request, clear_data))
        tracked_count = len(_TRACKED_SENDS)
        if not _TRACKED_SENDS_REAPER_ACTIVE:
            _TRACKED_SENDS_REAPER_ACTIVE = True
            thread = threading.Thread(
                target=_tracked_send_reaper_loop,
                name="GPUPreprocessReservationReaper",
                daemon=True,
            )
            thread.start()
            started = True
        else:
            started = False
    _trace(
        "tracker-register",
        request_id=_request_trace_id(request),
        tracker=hex(id(tracker)),
        tracked_count=tracked_count,
        reaper_started=started,
    )
    return True


def _tracked_send_reaper_loop() -> None:
    global _TRACKED_SENDS_REAPER_ACTIVE

    while True:
        ready: list[tuple[Any, Any, bool]] = []
        with _TRACKED_SENDS_LOCK:
            if not _TRACKED_SENDS:
                _TRACKED_SENDS_REAPER_ACTIVE = False
                return

            pending: list[tuple[Any, Any, bool]] = []
            for entry in _TRACKED_SENDS:
                tracker, request, clear_data = entry
                if getattr(tracker, "done", False):
                    ready.append(entry)
                else:
                    pending.append(entry)
            _TRACKED_SENDS[:] = pending

        for _tracker, request, clear_data in ready:
            _trace(
                "reaper-release",
                request_id=_request_trace_id(request),
                tracker=hex(id(_tracker)),
                ready=len(ready),
            )
            release_request_after_send(request, clear_data=clear_data)

        time.sleep(0.02)
