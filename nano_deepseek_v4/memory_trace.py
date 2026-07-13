from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import torch

MEMORY_TRACE_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {1, MEMORY_TRACE_SCHEMA_VERSION}
_EVENTS_FILENAME = "events.jsonl"
_MANIFEST_FILENAME = "manifest.json"
_BLOCK_ID_PATTERN = re.compile(r"^l[0-9]+:b[0-9]+:e[0-9]+$")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_phase(phase: str) -> None:
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unsupported memory trace phase: {phase!r}.")


@dataclass(frozen=True)
class MemoryTraceConfig:
    """Privacy-preserving identity fields for one native-memory trace."""

    trace_id: str
    request_id: str = "request-0"

    def __post_init__(self) -> None:
        for name, value in (("trace_id", self.trace_id), ("request_id", self.request_id)):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError(f"{name} must be a non-empty string of at most 256 characters.")


@dataclass(frozen=True)
class CacheMemoryAccounting:
    state_bytes: int
    hca_bytes: int
    csa_bytes: int
    index_bytes: int
    logical_cache_bytes: int
    hot_resident_bytes: int
    cold_resident_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_nonnegative_int(name, value)
        components = self.state_bytes + self.hca_bytes + self.csa_bytes + self.index_bytes
        if components != self.logical_cache_bytes:
            raise ValueError("cache component bytes must sum to logical_cache_bytes.")
        if self.hot_resident_bytes + self.cold_resident_bytes != self.logical_cache_bytes:
            raise ValueError("hot and cold resident bytes must sum to logical_cache_bytes.")


@dataclass(frozen=True)
class RankedBlock:
    block_id: str
    score: float

    def __post_init__(self) -> None:
        if _BLOCK_ID_PATTERN.fullmatch(self.block_id) is None:
            raise ValueError("ranked block has an invalid block ID.")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("ranked block score must be numeric.")
        if not isfinite(float(self.score)):
            raise ValueError("ranked block score must be finite.")


@dataclass(frozen=True)
class NativeSelection:
    batch_index: int
    query_position: int
    block_ids: tuple[str, ...]
    ranked_blocks: tuple[RankedBlock, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative_int("batch_index", self.batch_index)
        _require_nonnegative_int("query_position", self.query_position)
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("native selection block IDs must be unique.")
        if any(_BLOCK_ID_PATTERN.fullmatch(block_id) is None for block_id in self.block_ids):
            raise ValueError("native selection contains an invalid block ID.")
        ranked_ids = tuple(block.block_id for block in self.ranked_blocks)
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("ranked block IDs must be unique.")
        if not set(self.block_ids).issubset(ranked_ids) and self.ranked_blocks:
            raise ValueError("native selections must be present in ranked blocks.")


@dataclass(frozen=True)
class CSASelectionEvent:
    schema_version: int
    sequence_id: int
    event_type: Literal["csa_selection"]
    trace_id: str
    request_id: str
    layer_index: int
    phase: Literal["prefill", "decode"]
    logical_block_count: int
    block_bytes: int
    selections: tuple[NativeSelection, ...]
    indexer_wall_time_ns: int
    observer_wall_time_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS
            or self.event_type != "csa_selection"
        ):
            raise ValueError("invalid CSA selection event identity.")
        MemoryTraceConfig(trace_id=self.trace_id, request_id=self.request_id)
        _require_nonnegative_int("sequence_id", self.sequence_id)
        _require_nonnegative_int("layer_index", self.layer_index)
        _require_nonnegative_int("logical_block_count", self.logical_block_count)
        _require_nonnegative_int("block_bytes", self.block_bytes)
        _require_nonnegative_int("indexer_wall_time_ns", self.indexer_wall_time_ns)
        _require_nonnegative_int("observer_wall_time_ns", self.observer_wall_time_ns)
        _require_phase(self.phase)
        if any(
            len(selection.block_ids) > self.logical_block_count for selection in self.selections
        ):
            raise ValueError("native selection exceeds the logical block count.")
        if any(
            len(selection.ranked_blocks) > self.logical_block_count for selection in self.selections
        ):
            raise ValueError("ranked blocks exceed the logical block count.")
        if self.schema_version == 1 and (
            self.block_bytes != 0 or any(selection.ranked_blocks for selection in self.selections)
        ):
            raise ValueError("schema v1 events cannot contain replay metadata.")


@dataclass(frozen=True)
class CacheAdvanceEvent:
    schema_version: int
    sequence_id: int
    event_type: Literal["cache_advance"]
    trace_id: str
    request_id: str
    phase: Literal["prefill", "decode"]
    tokens_added: int
    seen_tokens_before: int
    seen_tokens_after: int
    accounting: CacheMemoryAccounting
    host_to_device_bytes: int
    device_to_host_bytes: int
    observer_wall_time_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS
            or self.event_type != "cache_advance"
        ):
            raise ValueError("invalid cache advance event identity.")
        MemoryTraceConfig(trace_id=self.trace_id, request_id=self.request_id)
        for name in (
            "sequence_id",
            "tokens_added",
            "seen_tokens_before",
            "seen_tokens_after",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "observer_wall_time_ns",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        _require_phase(self.phase)
        if self.tokens_added == 0:
            raise ValueError("tokens_added must be positive.")
        if self.seen_tokens_after != self.seen_tokens_before + self.tokens_added:
            raise ValueError("cache advance token counts are inconsistent.")


MemoryTraceEvent = CSASelectionEvent | CacheAdvanceEvent


@dataclass(frozen=True)
class MemoryTraceManifest:
    schema_version: int
    trace_id: str
    request_id: str
    event_count: int
    events_filename: str
    events_sha256: str
    contains_raw_text: bool

    def __post_init__(self) -> None:
        if self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported memory trace manifest schema version.")
        MemoryTraceConfig(trace_id=self.trace_id, request_id=self.request_id)
        _require_nonnegative_int("event_count", self.event_count)
        if self.events_filename != _EVENTS_FILENAME:
            raise ValueError("memory trace manifest has an invalid events filename.")
        if re.fullmatch(r"[0-9a-f]{64}", self.events_sha256) is None:
            raise ValueError("memory trace manifest has an invalid SHA-256 digest.")
        if self.contains_raw_text is not False:
            raise ValueError("memory trace manifest violates the M0 privacy contract.")


@dataclass(frozen=True)
class MemoryTraceResult:
    manifest: MemoryTraceManifest
    events: tuple[MemoryTraceEvent, ...]


@dataclass(frozen=True)
class NativeSelectionReplay:
    layer_index: int
    batch_index: int
    query_position: int
    block_ids: tuple[str, ...]


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _dict_nbytes(values: dict[str, torch.Tensor | None], name: str | None = None) -> int:
    if name is None:
        return sum(_tensor_nbytes(tensor) for tensor in values.values())
    return _tensor_nbytes(values.get(name))


def measure_cache_memory(cache: Any) -> CacheMemoryAccounting:
    """Measure tensors held by a DeepSeekV4Cache without reading tensor values."""

    state_bytes = 0
    hca_bytes = 0
    csa_bytes = 0
    index_bytes = 0
    dictionary_attributes = (
        "buffer_kv",
        "buffer_gate",
        "buffer_positions",
        "history_kv",
        "history_gate",
        "history_positions",
        "compressed_kv",
        "compressed_positions",
        "overlap_kv",
        "overlap_gate",
        "overlap_positions",
    )
    layer_types = cache.config.layer_types
    if layer_types is None or len(layer_types) != len(cache.layers):
        raise ValueError("cache configuration has an invalid layer schedule.")

    for layer, layer_type in zip(cache.layers, layer_types, strict=True):
        state_bytes += _tensor_nbytes(layer.local_kv)
        state_bytes += _tensor_nbytes(layer.local_positions)
        for attribute in dictionary_attributes:
            values = getattr(layer, attribute)
            index_bytes += _dict_nbytes(values, "indexer")
            compressor_bytes = _dict_nbytes(values, "compressor")
            if layer_type == "heavily_compressed_attention":
                hca_bytes += compressor_bytes
            elif layer_type == "compressed_sparse_attention":
                csa_bytes += compressor_bytes

    logical = state_bytes + hca_bytes + csa_bytes + index_bytes
    return CacheMemoryAccounting(
        state_bytes=state_bytes,
        hca_bytes=hca_bytes,
        csa_bytes=csa_bytes,
        index_bytes=index_bytes,
        logical_cache_bytes=logical,
        hot_resident_bytes=logical,
    )


def measure_csa_block_bytes(layer_cache: Any, logical_block_count: int) -> int:
    """Return the resident CSA compressor+indexer bytes for one batch-local block."""

    _require_nonnegative_int("logical_block_count", logical_block_count)
    if logical_block_count == 0:
        return 0
    tensors: list[torch.Tensor | None] = []
    for attribute in ("compressed_kv", "compressed_positions"):
        values = getattr(layer_cache, attribute)
        tensors.extend(values.get(name) for name in ("compressor", "indexer"))
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return 0
    batch_size = int(present[0].shape[0])
    if batch_size <= 0:
        raise ValueError("CSA cache tensors must have a positive batch dimension.")
    for tensor in present:
        if tensor.ndim < 2 or tuple(tensor.shape[:2]) != (batch_size, logical_block_count):
            raise ValueError("CSA cache tensors do not match the logical block count.")
    total_bytes = sum(_tensor_nbytes(tensor) for tensor in present)
    slots = batch_size * logical_block_count
    if total_bytes % slots != 0:
        raise ValueError("CSA cache bytes are not uniform per logical block.")
    return total_bytes // slots


class AdaptiveMemoryTraceCollector:
    """Append-only M0 observer for native DeepSeek-V4 cache behavior.

    The collector never receives token IDs or prompt text and has no policy or
    cache-mutation API. It records native CSA selections and cache accounting.
    """

    def __init__(self, config: MemoryTraceConfig) -> None:
        self.config = config
        self._events: list[MemoryTraceEvent] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> tuple[MemoryTraceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def _append(self, event: MemoryTraceEvent) -> None:
        with self._lock:
            self._events.append(replace(event, sequence_id=len(self._events)))

    def record_csa_selection(
        self,
        *,
        layer_index: int,
        seen_tokens: int,
        query_positions: torch.Tensor,
        block_end_positions: torch.Tensor,
        sparse_mask: torch.Tensor | None,
        scores: torch.Tensor | None,
        block_bytes: int,
        indexer_wall_time_ns: int,
    ) -> None:
        started = perf_counter_ns()
        query_rows = query_positions.detach().to(device="cpu", dtype=torch.long).tolist()
        block_rows = block_end_positions.detach().to(device="cpu", dtype=torch.long).tolist()
        mask_rows = (
            sparse_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
            if sparse_mask is not None
            else None
        )
        score_rows = (
            scores.detach().to(device="cpu", dtype=torch.float32).tolist()
            if scores is not None
            else None
        )
        selections: list[NativeSelection] = []
        for batch_index, queries in enumerate(query_rows):
            blocks = block_rows[batch_index]
            for query_index, query_position in enumerate(queries):
                if mask_rows is None:
                    selected: tuple[str, ...] = ()
                else:
                    selected = tuple(
                        f"l{layer_index}:b{batch_index}:e{block_position}"
                        for block_position, keep in zip(
                            blocks,
                            mask_rows[batch_index][query_index],
                            strict=True,
                        )
                        if keep
                    )
                ranked_blocks: tuple[RankedBlock, ...] = ()
                if score_rows is not None:
                    candidates = [
                        (int(block_position), float(score))
                        for block_position, score in zip(
                            blocks,
                            score_rows[batch_index][query_index],
                            strict=True,
                        )
                        if isfinite(float(score))
                    ]
                    candidates.sort(key=lambda item: (-item[1], item[0]))
                    ranked_blocks = tuple(
                        RankedBlock(
                            block_id=f"l{layer_index}:b{batch_index}:e{block_position}",
                            score=score,
                        )
                        for block_position, score in candidates
                    )
                selections.append(
                    NativeSelection(
                        batch_index=batch_index,
                        query_position=int(query_position),
                        block_ids=selected,
                        ranked_blocks=ranked_blocks,
                    )
                )
        observer_wall_time_ns = perf_counter_ns() - started
        self._append(
            CSASelectionEvent(
                schema_version=MEMORY_TRACE_SCHEMA_VERSION,
                sequence_id=0,
                event_type="csa_selection",
                trace_id=self.config.trace_id,
                request_id=self.config.request_id,
                layer_index=layer_index,
                phase="prefill" if seen_tokens == 0 else "decode",
                logical_block_count=block_end_positions.shape[1],
                block_bytes=block_bytes,
                selections=tuple(selections),
                indexer_wall_time_ns=max(int(indexer_wall_time_ns), 0),
                observer_wall_time_ns=observer_wall_time_ns,
            )
        )

    def record_cache_advance(self, cache: Any, tokens_added: int, seen_tokens_before: int) -> None:
        started = perf_counter_ns()
        accounting = measure_cache_memory(cache)
        observer_wall_time_ns = perf_counter_ns() - started
        self._append(
            CacheAdvanceEvent(
                schema_version=MEMORY_TRACE_SCHEMA_VERSION,
                sequence_id=0,
                event_type="cache_advance",
                trace_id=self.config.trace_id,
                request_id=self.config.request_id,
                phase="prefill" if seen_tokens_before == 0 else "decode",
                tokens_added=int(tokens_added),
                seen_tokens_before=seen_tokens_before,
                seen_tokens_after=cache.seen_tokens,
                accounting=accounting,
                host_to_device_bytes=0,
                device_to_host_bytes=0,
                observer_wall_time_ns=observer_wall_time_ns,
            )
        )

    def result(self) -> MemoryTraceResult:
        events = self.events
        payload = _encode_events(events)
        manifest = _build_manifest(self.config, len(events), hashlib.sha256(payload).hexdigest())
        return MemoryTraceResult(manifest=manifest, events=events)

    def write(self, trace_dir: str | Path) -> MemoryTraceResult:
        """Atomically publish JSONL events followed by their digest manifest."""

        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        events = self.events
        payload = _encode_events(events)
        events_sha256 = hashlib.sha256(payload).hexdigest()
        manifest = _build_manifest(self.config, len(events), events_sha256)
        events_temp: Path | None = None
        manifest_temp: Path | None = None
        try:
            events_temp = _write_temporary(trace_dir, ".memory-events-", payload)
            manifest_payload = (
                json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
            ).encode()
            manifest_temp = _write_temporary(trace_dir, ".memory-manifest-", manifest_payload)
            os.replace(events_temp, trace_dir / _EVENTS_FILENAME)
            events_temp = None
            os.replace(manifest_temp, trace_dir / _MANIFEST_FILENAME)
            manifest_temp = None
        finally:
            if events_temp is not None:
                events_temp.unlink(missing_ok=True)
            if manifest_temp is not None:
                manifest_temp.unlink(missing_ok=True)
        return MemoryTraceResult(manifest=manifest, events=events)


def _build_manifest(
    config: MemoryTraceConfig, event_count: int, digest: str
) -> MemoryTraceManifest:
    return MemoryTraceManifest(
        schema_version=MEMORY_TRACE_SCHEMA_VERSION,
        trace_id=config.trace_id,
        request_id=config.request_id,
        event_count=event_count,
        events_filename=_EVENTS_FILENAME,
        events_sha256=digest,
        contains_raw_text=False,
    )


def _write_temporary(directory: Path, prefix: str, payload: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=prefix, suffix=".tmp", delete=False
    ) as handle:
        path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _encode_events(events: tuple[MemoryTraceEvent, ...]) -> bytes:
    return b"".join(
        (json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n").encode()
        for event in events
    )


def _require_exact_keys(data: dict[str, Any], expected: set[str], label: str) -> None:
    if set(data) != expected:
        raise ValueError(f"{label} fields do not match schema v{MEMORY_TRACE_SCHEMA_VERSION}.")


def _parse_ranked_block(data: dict[str, Any]) -> RankedBlock:
    _require_exact_keys(data, {"block_id", "score"}, "ranked block")
    return RankedBlock(block_id=str(data["block_id"]), score=float(data["score"]))


def _parse_selection(data: dict[str, Any], schema_version: int) -> NativeSelection:
    expected = {"batch_index", "query_position", "block_ids"}
    if schema_version >= 2:
        expected.add("ranked_blocks")
    _require_exact_keys(data, expected, "selection")
    return NativeSelection(
        batch_index=int(data["batch_index"]),
        query_position=int(data["query_position"]),
        block_ids=tuple(str(block_id) for block_id in data["block_ids"]),
        ranked_blocks=tuple(_parse_ranked_block(item) for item in data.get("ranked_blocks", [])),
    )


def _parse_event(data: dict[str, Any]) -> MemoryTraceEvent:
    schema_version = data.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("memory trace event has an unsupported schema version.")
    event_type = data.get("event_type")
    if event_type == "csa_selection":
        expected = {
            "schema_version",
            "sequence_id",
            "event_type",
            "trace_id",
            "request_id",
            "layer_index",
            "phase",
            "logical_block_count",
            "selections",
            "indexer_wall_time_ns",
            "observer_wall_time_ns",
        }
        if schema_version >= 2:
            expected.add("block_bytes")
        _require_exact_keys(data, expected, "CSA selection event")
        return CSASelectionEvent(
            schema_version=int(data["schema_version"]),
            sequence_id=int(data["sequence_id"]),
            event_type="csa_selection",
            trace_id=str(data["trace_id"]),
            request_id=str(data["request_id"]),
            layer_index=int(data["layer_index"]),
            phase=data["phase"],
            logical_block_count=int(data["logical_block_count"]),
            block_bytes=int(data.get("block_bytes", 0)),
            selections=tuple(
                _parse_selection(item, int(schema_version)) for item in data["selections"]
            ),
            indexer_wall_time_ns=int(data["indexer_wall_time_ns"]),
            observer_wall_time_ns=int(data["observer_wall_time_ns"]),
        )
    if event_type == "cache_advance":
        _require_exact_keys(
            data,
            {
                "schema_version",
                "sequence_id",
                "event_type",
                "trace_id",
                "request_id",
                "phase",
                "tokens_added",
                "seen_tokens_before",
                "seen_tokens_after",
                "accounting",
                "host_to_device_bytes",
                "device_to_host_bytes",
                "observer_wall_time_ns",
            },
            "cache advance event",
        )
        accounting = CacheMemoryAccounting(**data["accounting"])
        return CacheAdvanceEvent(
            schema_version=int(data["schema_version"]),
            sequence_id=int(data["sequence_id"]),
            event_type="cache_advance",
            trace_id=str(data["trace_id"]),
            request_id=str(data["request_id"]),
            phase=data["phase"],
            tokens_added=int(data["tokens_added"]),
            seen_tokens_before=int(data["seen_tokens_before"]),
            seen_tokens_after=int(data["seen_tokens_after"]),
            accounting=accounting,
            host_to_device_bytes=int(data["host_to_device_bytes"]),
            device_to_host_bytes=int(data["device_to_host_bytes"]),
            observer_wall_time_ns=int(data["observer_wall_time_ns"]),
        )
    raise ValueError(f"unknown memory trace event type: {event_type!r}.")


def load_memory_trace(trace_dir: str | Path) -> MemoryTraceResult:
    """Load a trace only after validating schema, event count, and SHA-256."""

    trace_dir = Path(trace_dir)
    try:
        manifest_data = json.loads((trace_dir / _MANIFEST_FILENAME).read_text(encoding="utf-8"))
        if not isinstance(manifest_data, dict):
            raise ValueError("memory trace manifest must be an object.")
        _require_exact_keys(
            manifest_data,
            {
                "schema_version",
                "trace_id",
                "request_id",
                "event_count",
                "events_filename",
                "events_sha256",
                "contains_raw_text",
            },
            "manifest",
        )
        manifest = MemoryTraceManifest(**manifest_data)
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("memory trace manifest is missing or invalid.") from exc
    payload = (trace_dir / manifest.events_filename).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.events_sha256:
        raise ValueError("memory trace event checksum does not match its manifest.")
    events: list[MemoryTraceEvent] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            data = json.loads(line)
            event = _parse_event(data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid memory trace event on line {line_number}.") from exc
        if event.sequence_id != len(events):
            raise ValueError("memory trace sequence IDs are not contiguous.")
        if event.schema_version != manifest.schema_version:
            raise ValueError("memory trace event schema does not match its manifest.")
        if event.trace_id != manifest.trace_id or event.request_id != manifest.request_id:
            raise ValueError("memory trace event identity does not match its manifest.")
        events.append(event)
    if len(events) != manifest.event_count:
        raise ValueError("memory trace event count does not match its manifest.")
    return MemoryTraceResult(manifest=manifest, events=tuple(events))


def replay_native_selected_sets(
    trace: MemoryTraceResult | str | Path,
) -> tuple[NativeSelectionReplay, ...]:
    """Reconstruct the native CSA block set selected for every traced query."""

    result = load_memory_trace(trace) if isinstance(trace, (str, Path)) else trace
    replay: list[NativeSelectionReplay] = []
    for event in result.events:
        if not isinstance(event, CSASelectionEvent):
            continue
        replay.extend(
            NativeSelectionReplay(
                layer_index=event.layer_index,
                batch_index=selection.batch_index,
                query_position=selection.query_position,
                block_ids=selection.block_ids,
            )
            for selection in event.selections
        )
    return tuple(replay)
