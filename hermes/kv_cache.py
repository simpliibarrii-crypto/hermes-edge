"""Key/value cache managers for Hermes incremental decoding.

Three cache strategies are provided, trading off memory, context length, and
multi-session serving:

* :class:`StaticKVCache` — a single pre-allocated, fixed-size cache. This is the
  shape LiteRT-LM exports on device: the converted TFLite ``decode`` signature
  writes into a buffer sized to ``max_seq_len`` and never reallocates.
* :class:`SlidingWindowKVCache` — a :class:`StaticKVCache` subclass that keeps a
  rolling window of the most recent ``window_size`` tokens, evicting the oldest
  when full. Lets a 4096-ctx model hold an arbitrarily long conversation at a
  bounded memory cost (at the price of forgetting distant context).
* :class:`PagedKVCache` — block-level paging à la vLLM. The cache is carved into
  fixed-size blocks that are allocated on demand and freed per sequence, which
  makes it suitable for serving many concurrent sessions from one pool.

All caches expose ``to(device)`` for device migration and
``state_dict``/``load_state_dict`` for clean (de)serialization.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

KVTuple = Tuple[torch.Tensor, torch.Tensor]


class StaticKVCache:
    """Pre-allocated fixed-size KV cache (the LiteRT-LM on-device shape).

    Each layer owns a ``[1, num_kv_heads, max_seq_len, head_dim]`` key and value
    buffer. :meth:`update` writes new entries at ``position`` and returns the
    valid prefix; :meth:`get` returns the currently-filled slice.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        batch_size: int = 1,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._len = 0
        self._alloc()

    def _alloc(self) -> None:
        shape = (self.batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim)
        self.keys: List[torch.Tensor] = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.values: List[torch.Tensor] = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

    @property
    def current_len(self) -> int:
        """Number of valid (written) timesteps in the cache."""
        return self._len

    def reset(self) -> None:
        """Zero the fill pointer (buffers are reused, not reallocated)."""
        self._len = 0

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, position: int
    ) -> KVTuple:
        """Write ``k``/``v`` at ``position`` and return the valid prefix.

        Args:
            layer_idx: Which decoder layer this cache slot belongs to.
            k: Keys ``[B, num_kv_heads, T_new, head_dim]``.
            v: Values with the same shape.
            position: Start timestep to write at (the current sequence length).

        Returns:
            ``(keys, values)`` covering timesteps ``[0, position + T_new)``.

        Raises:
            ValueError: If the write would exceed ``max_seq_len``.
        """
        t_new = k.shape[2]
        end = position + t_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: writing {t_new} tokens at position "
                f"{position} exceeds max_seq_len={self.max_seq_len}."
            )
        self.keys[layer_idx][:, :, position:end, :] = k.to(self.dtype)
        self.values[layer_idx][:, :, position:end, :] = v.to(self.dtype)
        # The fill pointer tracks the furthest layer write of the current step.
        self._len = max(self._len, end)
        return self.get(layer_idx)

    def get(self, layer_idx: int) -> KVTuple:
        """Return the valid ``(keys, values)`` prefix for ``layer_idx``."""
        return (
            self.keys[layer_idx][:, :, : self._len, :],
            self.values[layer_idx][:, :, : self._len, :],
        )

    def to(self, device: torch.device | str) -> "StaticKVCache":
        """Move all buffers to ``device`` in place and return self."""
        self.device = torch.device(device)
        self.keys = [k.to(self.device) for k in self.keys]
        self.values = [v.to(self.device) for v in self.values]
        return self

    def state_dict(self) -> Dict[str, object]:
        """Serialize cache contents + metadata into a plain dict."""
        return {
            "class": type(self).__name__,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "max_seq_len": self.max_seq_len,
            "head_dim": self.head_dim,
            "batch_size": self.batch_size,
            "len": self._len,
            "keys": [k.cpu() for k in self.keys],
            "values": [v.cpu() for v in self.values],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore cache contents from :meth:`state_dict` output."""
        self._len = int(state["len"])  # type: ignore[arg-type]
        self.keys = [k.to(self.device) for k in state["keys"]]  # type: ignore[union-attr]
        self.values = [v.to(self.device) for v in state["values"]]  # type: ignore[union-attr]


class SlidingWindowKVCache(StaticKVCache):
    """Rolling-window KV cache that evicts the oldest tokens when full.

    Keeps at most ``window_size`` timesteps. Once full, each new write shifts the
    buffer left (dropping the oldest entries) so memory stays bounded — handy for
    long-running chats inside the 4096-token model.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        batch_size: int = 1,
        window_size: int = 1024,
    ) -> None:
        self.window_size = min(window_size, max_seq_len)
        super().__init__(
            num_layers, num_kv_heads, max_seq_len, head_dim, dtype, device, batch_size
        )

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, position: int
    ) -> KVTuple:
        t_new = k.shape[2]
        if t_new > self.window_size:
            # Only the most recent window_size tokens can be retained.
            k = k[:, :, -self.window_size :, :]
            v = v[:, :, -self.window_size :, :]
            t_new = self.window_size

        if self._len + t_new > self.window_size:
            evict = self._len + t_new - self.window_size
            kept = self._len - evict
            if kept > 0:
                self.keys[layer_idx][:, :, :kept, :] = self.keys[layer_idx][
                    :, :, evict : self._len, :
                ].clone()
                self.values[layer_idx][:, :, :kept, :] = self.values[layer_idx][
                    :, :, evict : self._len, :
                ].clone()
            write_at = kept
        else:
            write_at = self._len

        end = write_at + t_new
        self.keys[layer_idx][:, :, write_at:end, :] = k.to(self.dtype)
        self.values[layer_idx][:, :, write_at:end, :] = v.to(self.dtype)
        # current_len is shared across layers; cap it at the window.
        self._len = min(end, self.window_size)
        return self.get(layer_idx)


class PagedKVCache:
    """Block-paged KV cache for multi-session serving (vLLM-style).

    The KV pool is divided into ``num_blocks`` blocks of ``block_size`` tokens.
    Sequences request blocks via :meth:`allocate_block`; :meth:`free_sequence`
    returns a sequence's blocks to the free list. :meth:`get_page_table` exposes
    the per-sequence block mapping used to gather KV during attention.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int = 256,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)
        self._page_table: Dict[int, List[int]] = {}
        self._free: List[int] = list(range(num_blocks))
        self._alloc()

    def _alloc(self) -> None:
        # Pool shape: [num_layers, num_blocks, num_kv_heads, block_size, head_dim].
        shape = (
            self.num_layers,
            self.num_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )
        self.key_pool = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.value_pool = torch.zeros(shape, dtype=self.dtype, device=self.device)

    @property
    def num_free_blocks(self) -> int:
        """Count of blocks currently available for allocation."""
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        """Count of blocks currently assigned to some sequence."""
        return self.num_blocks - len(self._free)

    def allocate_block(self, seq_id: int = 0) -> int:
        """Pop a free block, assign it to ``seq_id``, and return its index.

        Raises:
            RuntimeError: If the pool is exhausted.
        """
        if not self._free:
            raise RuntimeError("PagedKVCache out of blocks; free a sequence first.")
        block = self._free.pop(0)
        self._page_table.setdefault(seq_id, []).append(block)
        return block

    def free_sequence(self, seq_id: int) -> List[int]:
        """Return all of ``seq_id``'s blocks to the free list.

        Returns the freed block indices (empty if the sequence was unknown).
        """
        blocks = self._page_table.pop(seq_id, [])
        self._free.extend(blocks)
        self._free.sort()
        return blocks

    def get_page_table(self) -> Dict[int, List[int]]:
        """Return a copy of the per-sequence ``seq_id -> [block_idx]`` mapping."""
        return {seq: list(blocks) for seq, blocks in self._page_table.items()}

    def reset(self) -> None:
        """Free every block and clear the page table."""
        self._page_table.clear()
        self._free = list(range(self.num_blocks))

    def to(self, device: torch.device | str) -> "PagedKVCache":
        """Move the KV pool to ``device`` in place and return self."""
        self.device = torch.device(device)
        self.key_pool = self.key_pool.to(self.device)
        self.value_pool = self.value_pool.to(self.device)
        return self

    def state_dict(self) -> Dict[str, object]:
        """Serialize pool contents + allocation state."""
        return {
            "class": type(self).__name__,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "page_table": self.get_page_table(),
            "free": list(self._free),
            "key_pool": self.key_pool.cpu(),
            "value_pool": self.value_pool.cpu(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore pool contents + allocation state from :meth:`state_dict`."""
        self._page_table = {
            int(k): list(v) for k, v in state["page_table"].items()  # type: ignore[union-attr]
        }
        self._free = list(state["free"])  # type: ignore[arg-type]
        self.key_pool = state["key_pool"].to(self.device)  # type: ignore[union-attr]
        self.value_pool = state["value_pool"].to(self.device)  # type: ignore[union-attr]
