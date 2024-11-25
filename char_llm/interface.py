import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
from torch import nn

_T = TypeVar('_T')


class Configurable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_config(self):
        pass

    @classmethod
    def from_config(cls: _T, config: dict) -> _T:
        return cls(**config)

    def save(self, path: str | Path) -> None:
        config = self.to_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls: _T, path: str | Path) -> _T:
        with open(path, 'r') as f:
            config = json.load(f)

        return cls.from_config(config)


@dataclass
class KVCache:
    max_size: int | None

    # (B, L)
    pos: torch.Tensor

    # (B, L, H)
    keys: torch.Tensor
    values: torch.Tensor

    def __post_init__(self):
        shape = self.values.shape
        assert self.keys.shape == shape
        assert self.pos.shape == shape[:2]

        batch_size, seq_length, hidden_size = shape

        if self.max_size is not None and seq_length > self.max_size:
            # evict the cache at the start
            to_evict = self.max_size - seq_length

            self.pos = self.pos[:, to_evict:]
            self.keys = self.keys[:, to_evict:]
            self.values = self.values[:, to_evict:]


@dataclass
class LayerCache:
    mem: KVCache
    window: KVCache


@dataclass
class ModelOutput:
    # B: batch size
    # L: sequence length
    # PA: "predict ahead" - number of tokens to predict in the future
    # V: size of the vocabulary
    # H: total hidden size of the model

    # model predicts the next PA characters
    logits: torch.Tensor  # (B, L, PA, V)

    # cache per layer
    cache: list[LayerCache]


class Generator(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
            self,
            tokens: torch.Tensor,
            absolute_pos: torch.Tensor,
            is_special_mask: torch.Tensor,
            past_keys: list[torch.Tensor] | None = None,
            past_values: list[torch.Tensor] | None = None,
            **kwargs
    ) -> ModelOutput:
        pass
