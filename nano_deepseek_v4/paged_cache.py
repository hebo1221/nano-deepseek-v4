from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PagedCacheAllocation:
    """Public snapshot of one request's logical KV-cache allocation."""

    request_id: str
    pages: tuple[int, ...]
    length: int
    token_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class PagedCacheStats:
    page_size: int
    total_pages: int
    free_pages: int
    used_pages: int
    active_requests: int
    resident_tokens: int


@dataclass
class _RequestPages:
    pages: list[int]
    length: int
    token_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class PagedKVCacheAllocator:
    """Reference block-table allocator for request-scoped paged KV caches.

    The allocator stores token-major tensors in fixed-size pages and exposes the
    page table for each request. It is deliberately a correctness reference for
    paged-cache semantics, not a GPU paged-attention implementation.
    """

    def __init__(self, page_size: int, num_pages: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive.")
        if num_pages <= 0:
            raise ValueError("num_pages must be positive.")
        self.page_size = int(page_size)
        self.num_pages = int(num_pages)
        self._free_pages = list(range(self.num_pages))
        self._requests: dict[str, _RequestPages] = {}
        self._page_storage: dict[int, torch.Tensor] = {}

    def append(self, request_id: str, values: torch.Tensor) -> PagedCacheAllocation:
        """Append token-major cache values to a request, allocating pages as needed."""

        self._validate_request_id(request_id)
        if values.ndim < 1:
            raise ValueError("values must be token-major with at least one dimension.")
        token_shape = tuple(values.shape[1:])
        entry = self._requests.get(request_id)
        if entry is not None:
            self._validate_compatible(entry, values)

        if values.shape[0] == 0:
            if entry is None:
                entry = _RequestPages(
                    pages=[],
                    length=0,
                    token_shape=token_shape,
                    dtype=values.dtype,
                    device=values.device,
                )
                self._requests[request_id] = entry
            return PagedCacheAllocation(
                request_id=request_id,
                pages=tuple(entry.pages),
                length=entry.length,
                token_shape=entry.token_shape,
                dtype=entry.dtype,
                device=entry.device,
            )

        if entry is not None:
            current_length = entry.length
            current_pages = len(entry.pages)
        else:
            current_length = 0
            current_pages = 0

        target_length = current_length + int(values.shape[0])
        required_pages = self._pages_for_length(target_length)
        pages_to_allocate = required_pages - current_pages
        if pages_to_allocate > len(self._free_pages):
            raise RuntimeError(
                f"Paged KV cache is out of pages: need {pages_to_allocate}, "
                f"only {len(self._free_pages)} free."
            )

        if entry is None:
            entry = _RequestPages(
                pages=[],
                length=0,
                token_shape=token_shape,
                dtype=values.dtype,
                device=values.device,
            )
            self._requests[request_id] = entry

        for _ in range(pages_to_allocate):
            page_id = self._allocate_page()
            entry.pages.append(page_id)
            self._page_storage[page_id] = values.new_zeros((self.page_size, *token_shape))

        offset = 0
        total_tokens = int(values.shape[0])
        while offset < total_tokens:
            page_index = entry.length // self.page_size
            page_offset = entry.length % self.page_size
            take = min(self.page_size - page_offset, total_tokens - offset)
            page_id = entry.pages[page_index]
            self._page_storage[page_id][page_offset : page_offset + take].copy_(
                values[offset : offset + take]
            )
            entry.length += take
            offset += take

        return self.allocation(request_id)

    def read(self, request_id: str) -> torch.Tensor:
        """Materialize a request's logical cache sequence from its page table."""

        entry = self._require_request(request_id)
        if entry.length == 0:
            return torch.empty((0, *entry.token_shape), dtype=entry.dtype, device=entry.device)

        chunks = []
        remaining = entry.length
        for page_id in entry.pages:
            take = min(self.page_size, remaining)
            chunks.append(self._page_storage[page_id][:take])
            remaining -= take
            if remaining == 0:
                break
        return torch.cat(chunks, dim=0)

    def crop(self, request_id: str, length: int) -> PagedCacheAllocation:
        """Shorten a request and free pages beyond the requested logical length."""

        entry = self._require_request(request_id)
        if length < 0:
            raise ValueError("length must be non-negative.")
        if length > entry.length:
            raise ValueError(f"Cannot crop request {request_id!r} from {entry.length} to {length}.")

        required_pages = self._pages_for_length(length)
        for page_id in entry.pages[required_pages:]:
            self._free_page(page_id)
        del entry.pages[required_pages:]
        entry.length = int(length)

        tail = entry.length % self.page_size
        if tail and entry.pages:
            self._page_storage[entry.pages[-1]][tail:].zero_()

        return self.allocation(request_id)

    def evict(self, request_id: str) -> None:
        """Free all pages for a request and remove its page table."""

        entry = self._require_request(request_id)
        for page_id in entry.pages:
            self._free_page(page_id)
        del self._requests[request_id]

    def page_table(self, request_id: str) -> list[int]:
        return list(self._require_request(request_id).pages)

    def allocation(self, request_id: str) -> PagedCacheAllocation:
        entry = self._require_request(request_id)
        return PagedCacheAllocation(
            request_id=request_id,
            pages=tuple(entry.pages),
            length=entry.length,
            token_shape=entry.token_shape,
            dtype=entry.dtype,
            device=entry.device,
        )

    def page_tensor(self, page_id: int) -> torch.Tensor:
        if page_id not in self._page_storage:
            raise KeyError(f"Page {page_id} is not allocated.")
        return self._page_storage[page_id]

    def stats(self) -> PagedCacheStats:
        return PagedCacheStats(
            page_size=self.page_size,
            total_pages=self.num_pages,
            free_pages=len(self._free_pages),
            used_pages=self.num_pages - len(self._free_pages),
            active_requests=len(self._requests),
            resident_tokens=sum(entry.length for entry in self._requests.values()),
        )

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string.")

    def _pages_for_length(self, length: int) -> int:
        length = int(length)
        if length < 0:
            raise ValueError("length must be non-negative.")
        if length == 0:
            return 0
        return (length + self.page_size - 1) // self.page_size

    def _allocate_page(self) -> int:
        page_id = self._free_pages.pop(0)
        return page_id

    def _free_page(self, page_id: int) -> None:
        self._page_storage.pop(page_id, None)
        self._free_pages.append(page_id)
        self._free_pages.sort()

    def _require_request(self, request_id: str) -> _RequestPages:
        self._validate_request_id(request_id)
        if request_id not in self._requests:
            raise KeyError(f"Unknown request_id: {request_id!r}")
        return self._requests[request_id]

    @staticmethod
    def _validate_compatible(entry: _RequestPages, values: torch.Tensor) -> None:
        if tuple(values.shape[1:]) != entry.token_shape:
            raise ValueError(
                f"Cache value shape mismatch: expected token shape {entry.token_shape}, "
                f"got {tuple(values.shape[1:])}."
            )
        if values.dtype != entry.dtype:
            raise ValueError(f"Cache dtype mismatch: expected {entry.dtype}, got {values.dtype}.")
        if values.device != entry.device:
            raise ValueError(f"Cache device mismatch: expected {entry.device}, got {values.device}.")
