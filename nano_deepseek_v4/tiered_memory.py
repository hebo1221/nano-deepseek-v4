from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass(frozen=True)
class TieredMemoryStats:
    logical_blocks: int
    hot_blocks: int
    protected_blocks: int
    logical_bytes: int
    hot_bytes: int
    host_bytes: int
    h2d_bytes: int
    d2h_bytes: int
    h2d_count: int
    d2h_count: int
    useful_h2d_bytes: int
    late_misses: int
    prefetches: int
    evictions: int

    @property
    def useful_h2d_ratio(self) -> float:
        if self.h2d_bytes == 0:
            return 1.0
        return self.useful_h2d_bytes / self.h2d_bytes


class TieredBlockStore:
    """Reference GPU-hot/CPU-cold store for immutable CSA value blocks.

    The CPU tensor is the canonical logical cache. A bounded set of blocks is
    copied to the target device and replaced on every prefetch. CUDA copies use
    a dedicated stream and are joined only when ``resolve`` consumes them.
    """

    def __init__(
        self,
        host_values: torch.Tensor,
        host_positions: torch.Tensor,
        *,
        hot_budget_blocks: int,
        device: torch.device | str,
        protected_blocks: Iterable[int] = (),
        async_transfer: bool = True,
        initial_hot_blocks: Iterable[int] = (),
    ) -> None:
        if host_values.ndim != 3 or host_positions.ndim != 2:
            raise ValueError("Tiered blocks require [batch, blocks, dim] values and positions.")
        if host_values.shape[:2] != host_positions.shape:
            raise ValueError("Tiered value and position shapes are inconsistent.")
        if (
            isinstance(hot_budget_blocks, bool)
            or not isinstance(hot_budget_blocks, int)
            or hot_budget_blocks <= 0
        ):
            raise ValueError("hot_budget_blocks must be a positive integer.")
        if not isinstance(async_transfer, bool):
            raise ValueError("async_transfer must be boolean.")
        if host_values.device.type != "cpu" or host_positions.device.type != "cpu":
            raise ValueError("Canonical tiered tensors must reside on CPU.")

        self.hot_budget_blocks = hot_budget_blocks
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA tiering requested but CUDA is unavailable.")
        self.async_transfer = async_transfer and self.device.type == "cuda"
        self.host_values = self._canonical_host_copy(host_values)
        self.host_positions = self._canonical_host_copy(host_positions)
        self.protected_blocks = self._normalize_indices(protected_blocks)
        if len(self.protected_blocks) > self.hot_budget_blocks:
            raise ValueError("Protected blocks exceed the hot-block budget.")

        self._stream = torch.cuda.Stream(device=self.device) if self.async_transfer else None
        self._pending_event: torch.cuda.Event | None = None
        self._pending_staging: tuple[torch.Tensor, torch.Tensor] | None = None
        self._hot_values: torch.Tensor | None = None
        self._hot_positions: torch.Tensor | None = None
        self._hot_indices: tuple[int, ...] = ()
        self._h2d_bytes = 0
        self._d2h_bytes = 0
        self._h2d_count = 0
        self._d2h_count = 0
        self._useful_h2d_bytes = 0
        self._late_misses = 0
        self._prefetches = 0
        self._evictions = 0

        initial = tuple(initial_hot_blocks)
        if initial or self.protected_blocks:
            self.prefetch(initial)
            self._wait_pending()

    @classmethod
    def from_device_tensors(
        cls,
        values: torch.Tensor,
        positions: torch.Tensor,
        *,
        hot_budget_blocks: int,
        device: torch.device | str | None = None,
        protected_blocks: Iterable[int] = (),
        async_transfer: bool = True,
        initial_hot_blocks: Iterable[int] = (),
    ) -> TieredBlockStore:
        target = values.device if device is None else torch.device(device)
        use_pinned = target.type == "cuda" and torch.cuda.is_available()
        host_values = torch.empty_like(values, device="cpu", pin_memory=use_pinned)
        host_positions = torch.empty_like(positions, device="cpu", pin_memory=use_pinned)
        non_blocking = use_pinned and values.device.type == "cuda"
        host_values.copy_(values, non_blocking=non_blocking)
        host_positions.copy_(positions, non_blocking=non_blocking)
        if non_blocking:
            torch.cuda.current_stream(values.device).synchronize()
        store = cls(
            host_values,
            host_positions,
            hot_budget_blocks=hot_budget_blocks,
            device=target,
            protected_blocks=protected_blocks,
            async_transfer=async_transfer,
            initial_hot_blocks=initial_hot_blocks,
        )
        if values.device.type == "cuda":
            store._d2h_bytes += _tensor_bytes(values) + _tensor_bytes(positions)
            store._d2h_count += 1
        return store

    @property
    def batch_size(self) -> int:
        return int(self.host_values.shape[0])

    @property
    def num_blocks(self) -> int:
        return int(self.host_values.shape[1])

    @property
    def hot_indices(self) -> tuple[int, ...]:
        return self._hot_indices

    def transfer_state(self) -> dict[str, int]:
        """Return cumulative transfer/runtime counters for continuation checkpoints."""

        return {
            "h2d_bytes": self._h2d_bytes,
            "d2h_bytes": self._d2h_bytes,
            "h2d_count": self._h2d_count,
            "d2h_count": self._d2h_count,
            "useful_h2d_bytes": self._useful_h2d_bytes,
            "late_misses": self._late_misses,
            "prefetches": self._prefetches,
            "evictions": self._evictions,
        }

    def continue_transfer_state(self, prior: dict[str, int]) -> None:
        """Continue a prior counter epoch while retaining materialization traffic.

        Constructing a clone or restoring a checkpoint can itself materialize hot
        blocks.  Those new counters are added to, rather than substituted for,
        the prior history so cumulative physical telemetry never regresses.
        """

        current = self.transfer_state()
        if set(prior) != set(current):
            raise ValueError("Tiered transfer state has an invalid counter schema.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in prior.values()
        ):
            raise ValueError("Tiered transfer counters must be non-negative integers.")
        if prior["useful_h2d_bytes"] > prior["h2d_bytes"]:
            raise ValueError("Useful H2D bytes cannot exceed total H2D bytes.")
        for name, value in prior.items():
            setattr(self, f"_{name}", value + current[name])

    def hot_end_positions(self, batch_index: int = 0) -> tuple[int, ...]:
        """Return the logical end-position identities materialized in the hot tier."""

        if (
            isinstance(batch_index, bool)
            or not isinstance(batch_index, int)
            or not 0 <= batch_index < self.batch_size
        ):
            raise IndexError("Tiered batch index is out of range.")
        return tuple(
            int(self.host_positions[batch_index, block_index]) for block_index in self._hot_indices
        )

    def _canonical_host_copy(self, tensor: torch.Tensor) -> torch.Tensor:
        pin = self.device.type == "cuda" and torch.cuda.is_available()
        result = torch.empty_like(tensor, device="cpu", pin_memory=pin)
        result.copy_(tensor)
        return result

    def _normalize_indices(self, indices: Iterable[int]) -> tuple[int, ...]:
        raw = tuple(indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in raw):
            raise ValueError("Tiered block indices must be integers.")
        result = tuple(sorted(set(raw)))
        if any(index < 0 or index >= self.num_blocks for index in result):
            raise IndexError("Tiered block index is out of range.")
        return result

    def _block_bytes(self) -> int:
        if self.num_blocks == 0:
            return 0
        return (
            _tensor_bytes(self.host_values) + _tensor_bytes(self.host_positions)
        ) // self.num_blocks

    def _wait_pending(self) -> None:
        if self._pending_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._pending_event)
            self._pending_event.synchronize()
            self._pending_event = None
        self._pending_staging = None

    def synchronize(self) -> None:
        """Wait until any pending hot-tier transfer is physically complete."""

        self._wait_pending()

    def fill_hot_budget(self) -> None:
        """Materialize the exact configured capacity without outcome-derived inputs.

        Existing and protected residents are retained first. Any newly available
        slots are filled by ascending logical block index, which makes an upward
        quota resize deterministic and gives resident-only recovery an exact-size
        physical set before the next query arrives.
        """

        if self.num_blocks < self.hot_budget_blocks:
            raise RuntimeError(
                "Logical tier candidates cannot exact-fill the configured hot budget."
            )
        retained = tuple(dict.fromkeys((*self.protected_blocks, *self._hot_indices)))
        selected = list(retained[: self.hot_budget_blocks])
        selected_set = set(selected)
        for block_index in range(self.num_blocks):
            if len(selected) == self.hot_budget_blocks:
                break
            if block_index not in selected_set:
                selected.append(block_index)
                selected_set.add(block_index)
        self.prefetch(selected)

    def prefetch(self, indices: Iterable[int]) -> None:
        requested = self._normalize_indices(indices)
        target = tuple(sorted(set(requested) | set(self.protected_blocks)))
        if len(target) > self.hot_budget_blocks:
            raise RuntimeError(
                f"Requested {len(target)} hot blocks exceeds budget {self.hot_budget_blocks}."
            )
        self._wait_pending()
        if target == self._hot_indices:
            return

        previous = set(self._hot_indices)
        if previous - set(target):
            self._evictions += len(previous - set(target))
        index = torch.tensor(target, dtype=torch.long)
        staging_values = torch.index_select(self.host_values, 1, index)
        staging_positions = torch.index_select(self.host_positions, 1, index)
        if self.device.type == "cuda":
            if not staging_values.is_pinned():
                staging_values = staging_values.pin_memory()
            if not staging_positions.is_pinned():
                staging_positions = staging_positions.pin_memory()

        copied_bytes = len(target) * self._block_bytes()
        self._prefetches += 1

        # The canonical CPU tensors remain the recovery source, so the prior
        # device allocation is not needed while its replacement is copied.
        # Releasing it first prevents an old-plus-new HBM spike whenever a
        # quota rebalance changes the resident block set.  A transfer failure
        # therefore leaves an empty (recoverable) hot tier instead of silently
        # retaining a resident set that no longer matches ``target``.
        self._hot_values = None
        self._hot_positions = None
        self._hot_indices = ()
        if self.device.type == "cuda":
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    hot_values = staging_values.to(self.device, non_blocking=True)
                    hot_positions = staging_positions.to(self.device, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._stream)
                self._pending_event = event
                self._pending_staging = (staging_values, staging_positions)
            else:
                hot_values = staging_values.to(self.device)
                hot_positions = staging_positions.to(self.device)
            self._h2d_bytes += copied_bytes
            self._h2d_count += 1
            self._useful_h2d_bytes += len(requested) * self._block_bytes()
        else:
            hot_values = staging_values
            hot_positions = staging_positions
        self._hot_values = hot_values
        self._hot_positions = hot_positions
        self._hot_indices = target

    def resize_hot_budget(self, hot_budget_blocks: int) -> None:
        """Change the hot-block capacity without exceeding the new capacity.

        A shrinking store keeps protected blocks first, then the lowest-indexed
        already-hot blocks. Rebuilding happens only after the old hot tensors
        have been released, so a resize never internally retains both the old
        and replacement hot tensors at once.
        """

        if (
            isinstance(hot_budget_blocks, bool)
            or not isinstance(hot_budget_blocks, int)
            or hot_budget_blocks <= 0
        ):
            raise ValueError("hot_budget_blocks must be a positive integer.")
        if hot_budget_blocks < len(self.protected_blocks):
            raise ValueError("Protected blocks exceed the requested hot-block budget.")

        self._wait_pending()
        if hot_budget_blocks == self.hot_budget_blocks:
            return
        if len(self._hot_indices) <= hot_budget_blocks:
            self.hot_budget_blocks = hot_budget_blocks
            return

        protected = set(self.protected_blocks)
        remaining_slots = hot_budget_blocks - len(protected)
        retained_unprotected = tuple(
            index for index in self._hot_indices if index not in protected
        )[:remaining_slots]
        retained = tuple(sorted(protected | set(retained_unprotected)))
        evicted = len(set(self._hot_indices) - set(retained))

        # Drop the old device allocation before materializing its bounded
        # replacement. This avoids a transient old-plus-new hot allocation.
        self._hot_values = None
        self._hot_positions = None
        self._hot_indices = ()
        self.hot_budget_blocks = hot_budget_blocks
        self._evictions += evicted
        if retained:
            self.prefetch(retained)

    def resolve(self, indices: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
        requested = self._normalize_indices(indices)
        if not set(requested).issubset(self._hot_indices):
            self._late_misses += 1
            self.prefetch(requested)
        self._wait_pending()
        if self._hot_values is None or self._hot_positions is None:
            shape = (self.batch_size, 0, self.host_values.shape[2])
            return (
                torch.empty(shape, dtype=self.host_values.dtype, device=self.device),
                torch.empty(
                    (self.batch_size, 0), dtype=self.host_positions.dtype, device=self.device
                ),
            )
        lookup = {block: slot for slot, block in enumerate(self._hot_indices)}
        slots = torch.tensor([lookup[index] for index in requested], device=self.device)
        return (
            torch.index_select(self._hot_values, 1, slots),
            torch.index_select(self._hot_positions, 1, slots),
        )

    def append(self, values: torch.Tensor, positions: torch.Tensor) -> None:
        if values.ndim != 3 or positions.ndim != 2 or values.shape[:2] != positions.shape:
            raise ValueError("Appended tiered blocks have inconsistent shapes.")
        if values.shape[0] != self.batch_size or values.shape[2] != self.host_values.shape[2]:
            raise ValueError("Appended tiered blocks are incompatible with the store.")
        if values.shape[1] == 0:
            return
        added = type(self).from_device_tensors(
            values,
            positions,
            hot_budget_blocks=max(1, int(values.shape[1])),
            device=self.device,
            async_transfer=False,
        )
        self._wait_pending()
        combined_values = torch.cat([self.host_values, added.host_values], dim=1)
        combined_positions = torch.cat([self.host_positions, added.host_positions], dim=1)
        self.host_values = self._canonical_host_copy(combined_values)
        self.host_positions = self._canonical_host_copy(combined_positions)
        self._d2h_bytes += added._d2h_bytes
        self._d2h_count += added._d2h_count
        # Appending immutable logical blocks does not invalidate the identities
        # or device tensors of existing hot blocks. Keeping them resident is
        # required for a genuine resident-only fallback at compression
        # boundaries; the newly appended blocks remain cold until selected.

    def crop(self, max_position: int) -> None:
        if isinstance(max_position, bool) or not isinstance(max_position, int) or max_position < 0:
            raise ValueError("max_position must be a non-negative integer.")
        keep_by_batch = self.host_positions < max_position
        if self.batch_size > 1 and not all(
            torch.equal(keep_by_batch[batch_index], keep_by_batch[0])
            for batch_index in range(1, self.batch_size)
        ):
            raise ValueError("Tiered crop requires one shared logical position mask per batch.")
        keep = keep_by_batch[0] if self.batch_size else torch.zeros(0, dtype=torch.bool)
        kept = int(keep.sum().item())
        if not bool(keep[:kept].all()) or bool(keep[kept:].any()):
            raise ValueError("Tiered crop requires prefix-ordered logical block positions.")
        self._wait_pending()
        self.host_values = self._canonical_host_copy(self.host_values[:, keep])
        self.host_positions = self._canonical_host_copy(self.host_positions[:, keep])
        self.protected_blocks = tuple(
            index for index in self.protected_blocks if index < self.num_blocks
        )
        self._hot_values = None
        self._hot_positions = None
        self._hot_indices = ()

    def clone(self) -> TieredBlockStore:
        other = type(self)(
            self.host_values,
            self.host_positions,
            hot_budget_blocks=self.hot_budget_blocks,
            device=self.device,
            protected_blocks=self.protected_blocks,
            async_transfer=self.async_transfer,
            initial_hot_blocks=self._hot_indices,
        )
        other.continue_transfer_state(self.transfer_state())
        return other

    def select_batch(self, index: int) -> TieredBlockStore:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.batch_size
        ):
            raise IndexError("Tiered cache batch index is out of range.")
        other = type(self)(
            self.host_values[index : index + 1],
            self.host_positions[index : index + 1],
            hot_budget_blocks=self.hot_budget_blocks,
            device=self.device,
            protected_blocks=self.protected_blocks,
            async_transfer=self.async_transfer,
            initial_hot_blocks=self._hot_indices,
        )
        other.continue_transfer_state(self.transfer_state())
        return other

    @classmethod
    def stack(cls, stores: list[TieredBlockStore]) -> TieredBlockStore:
        if not stores:
            raise ValueError("Cannot stack an empty tiered-store list.")
        first = stores[0]
        if any(
            store.num_blocks != first.num_blocks
            or store.hot_budget_blocks != first.hot_budget_blocks
            or store.device != first.device
            or store.protected_blocks != first.protected_blocks
            or store.hot_indices != first.hot_indices
            or store.async_transfer != first.async_transfer
            or store.host_values.dtype != first.host_values.dtype
            or store.host_positions.dtype != first.host_positions.dtype
            or store.host_values.shape[1:] != first.host_values.shape[1:]
            or store.host_positions.shape[1:] != first.host_positions.shape[1:]
            for store in stores[1:]
        ):
            raise ValueError("Cannot stack incompatible tiered stores.")
        if first.batch_size == 0:
            raise ValueError("Cannot stack an empty-batch tiered store.")
        reference_positions = first.host_positions[0]
        if any(
            not torch.equal(store.host_positions[batch_index], reference_positions)
            for store in stores
            for batch_index in range(store.batch_size)
        ):
            raise ValueError(
                "Cannot stack tiered stores with different logical end-position schedules."
            )
        other = cls(
            torch.cat([store.host_values for store in stores], dim=0),
            torch.cat([store.host_positions for store in stores], dim=0),
            hot_budget_blocks=first.hot_budget_blocks,
            device=first.device,
            protected_blocks=first.protected_blocks,
            async_transfer=first.async_transfer,
            initial_hot_blocks=first._hot_indices,
        )
        combined = {
            name: sum(store.transfer_state()[name] for store in stores)
            for name in first.transfer_state()
        }
        other.continue_transfer_state(combined)
        return other

    def stats(self) -> TieredMemoryStats:
        hot_bytes = 0
        if self._hot_values is not None and self._hot_positions is not None:
            hot_bytes = _tensor_bytes(self._hot_values) + _tensor_bytes(self._hot_positions)
        host_bytes = _tensor_bytes(self.host_values) + _tensor_bytes(self.host_positions)
        return TieredMemoryStats(
            logical_blocks=self.num_blocks,
            hot_blocks=len(self._hot_indices),
            protected_blocks=len(self.protected_blocks),
            logical_bytes=host_bytes,
            hot_bytes=hot_bytes,
            host_bytes=host_bytes,
            h2d_bytes=self._h2d_bytes,
            d2h_bytes=self._d2h_bytes,
            h2d_count=self._h2d_count,
            d2h_count=self._d2h_count,
            useful_h2d_bytes=self._useful_h2d_bytes,
            late_misses=self._late_misses,
            prefetches=self._prefetches,
            evictions=self._evictions,
        )
