import abc
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type

import torch
from torch import nn

_T = TypeVar('_T')


class Configurable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_config(self):
        pass

    @classmethod
    def from_config(cls: Type[_T], config: dict, **extra_kwargs) -> _T:
        config = deepcopy(config)
        config.update(extra_kwargs)
        return cls(**config)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = self.to_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls: Type[_T], path: str | Path) -> _T:
        path = Path(path)
        assert path.exists() and path.is_file()

        with open(path, 'r') as f:
            config = json.load(f)

        return cls.from_config(config)


@dataclass
class Cache:  # FIXME: cache management is a bit complicated on non-batched tensors..
    """
    The input to our models is always a 1-D tensor.

    If you need to process a batch of inputs, make sure to concatenate them along sequence dimension
    via <EOS> token. The absolute position should be reset in each example.

    Memory management:
    The model will automatically create new memories every mem_freq tokens and store them to Cache.
    Make sure to pass `return_cache=True` to avoid recomputing states and memories.

    Types of attention:
    1. Local - dense attention on the nearest `max_local_distance` tokens.
    2. Global - sparse attention to memory tokens from the past.

    Global attention compresses the past, reducing excessive computation.

    How to work with cache:

    1. For each layer:
        1. Process new KV (project, embed position using `get_absolute_pos`)
        2. Save to list
        3. Get cached KV, calculate new Q
    2. Update cache and return if necessary
    """
    max_global_distance: int
    pos_local: torch.Tensor  # (L,)
    k_local: list[torch.Tensor]  # (L, H)
    v_local: list[torch.Tensor]  # (L, H)

    max_local_distance: int
    pos_mem: torch.Tensor  # (M)
    k_mem: list[torch.Tensor]  # (M, H)
    v_mem: list[torch.Tensor]  # (M, H)

    _dtype: torch.dtype
    _device: torch.device

    @classmethod
    def empty(
            cls,
            n_layers: int,
            hidden_dims: int,
            max_local_distance: int,
            max_global_distance: int,
            device: str | torch.device,
            state_dtype: torch.dtype
    ):
        assert max_local_distance > 0
        assert max_global_distance > 0

        return cls(
            max_global_distance=max_global_distance,
            max_local_distance=max_local_distance,
            pos_local=torch.zeros(0, dtype=torch.int, device=device),
            k_local=[torch.zeros((0, hidden_dims), dtype=state_dtype, device=device) for _ in range(n_layers)],
            v_local=[torch.zeros((0, hidden_dims), dtype=state_dtype, device=device) for _ in range(n_layers)],
            pos_mem=torch.zeros(0, dtype=torch.int, device=device),
            k_mem=[torch.zeros((0, hidden_dims), dtype=state_dtype, device=device) for _ in range(n_layers)],
            v_mem=[torch.zeros((0, hidden_dims), dtype=state_dtype, device=device) for _ in range(n_layers)],
            _dtype=state_dtype,
            _device=torch.device(device),
        )

    def to(self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        def move_to(device_or_dtype: torch.dtype | torch.device):
            self.v_local = [v.to(device_or_dtype) for v in self.v_local]
            self.k_local = [k.to(device_or_dtype) for k in self.k_local]

            self.v_mem = [v.to(device_or_dtype) for v in self.v_mem]
            self.k_mem = [k.to(device_or_dtype) for k in self.k_mem]

        if dtype is not None and dtype != self._dtype:
            self._dtype = dtype
            move_to(self._dtype)

        if device is not None:
            self._device = torch.device(device)
            move_to(self._device)

    def get_absolute_pos(self, relative_pos: torch.Tensor, *, is_mem: bool = False) -> torch.Tensor:
        # FIXME: only works when inferencing on single document!!!
        source_pos = self.pos_mem if is_mem else self.pos_local
        shift = source_pos[-1] + 1 if len(source_pos) else 0  # +1 for the cases like [0] mem and relative_pos [0,1]
        return relative_pos + shift

    def update(
            self,
            new_pos: torch.Tensor,
            new_v: list[torch.Tensor],
            new_k: list[torch.Tensor],
            *,
            is_mem: bool = False
    ):
        dest_k = self.k_mem if is_mem else self.k_local
        dest_v = self.v_mem if is_mem else self.v_local
        cut_at = self.max_global_distance if is_mem else self.max_local_distance

        assert len(new_v) == len(new_k) == len(dest_k) == len(dest_v)
        added_len = len(new_pos)

        for layer_idx, (k, v) in enumerate(zip(new_v, new_k)):
            assert added_len == len(v) == len(k)

            dest_k[layer_idx] = self._append_cut(dest_k[layer_idx], k, cut_at=cut_at)
            dest_v[layer_idx] = self._append_cut(dest_v[layer_idx], v, cut_at=cut_at)

        if is_mem:
            self.pos_mem = self._append_cut(self.pos_mem, new_pos, cut_at=cut_at)
        else:
            self.pos_local = self._append_cut(self.pos_local, new_pos, cut_at=cut_at)

    @staticmethod
    def _append_cut(dest: torch.Tensor, source: torch.Tensor, *, cut_at: int):
        source = source.to(dest.device, dtype=dest.dtype)
        return torch.concat((dest, source), dim=0)[-cut_at:]


@dataclass
class ModelOutput:
    # L: sequence length
    # V: size of the vocabulary
    # H: total hidden size of the model

    # model predicts the next PA characters
    logits: torch.Tensor  # (L, V)

    # cache per layer
    cache: Cache | None


class Generator(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
            self,
            tokens: torch.Tensor,
            past_cache: Cache | None = None,
            *,
            return_cache: bool = False
    ) -> ModelOutput:
        pass
