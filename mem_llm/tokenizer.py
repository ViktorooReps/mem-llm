import json
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TypeVar, Type

import tiktoken
import torch
from transformers import AutoTokenizer

from mem_llm.interface import ConfigurableMixin


TOKENIZER_CONFIG_FILENAME = 'tokenizer_config.json'

_T = TypeVar('_T')


class Tokenizer(ConfigurableMixin, metaclass=ABCMeta):
    TYPE = None

    @property
    def descriptor(self):
        # replace slashes since this can be used in file names
        return (self.TYPE + '-' + '-'.join(f'{k}{v}' for k, v in self.to_config().items())).replace('/', '_')
    
    def __len__(self) -> int:
        return self.size

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = self.to_config()
        with open(path / TOKENIZER_CONFIG_FILENAME, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls: Type[_T], path: str | Path, **extra_kwargs) -> _T:
        path = Path(path)
        if path.is_file():
            load_path = path
        elif path.is_dir():
            load_path = path / TOKENIZER_CONFIG_FILENAME
        else:
            raise ValueError(f'Path {path} does not exist')

        with open(load_path, 'r') as f:
            config = json.load(f)

        return cls.from_config(config, **extra_kwargs)

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str, *, add_eos: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        pass


class CharTokenizer(Tokenizer):
    TYPE = 'char'

    def __init__(self, vocab_size: int = 128, *, unk_token: int = 0, eos_token_id: int = 127):
        self._size = vocab_size

        self._unk_token = unk_token
        self._eos_token = eos_token_id

    @property
    def size(self) -> int:
        return self._size

    @property
    def unk_token(self) -> int:
        return self._unk_token

    @property
    def eos_token_id(self) -> int:
        return self._eos_token

    def to_config(self):
        return {
            'vocab_size': self.size,
            'unk_token': self.unk_token,
            'eos_token_id': self.eos_token_id
        }

    def encode(self, text: str, *, add_eos: bool = False) -> torch.Tensor:
        tokens = list(map(ord, text))
        if add_eos:
            tokens.append(self.eos_token_id)

        tokens_torch = torch.tensor(tokens, dtype=torch.long)
        tokens_torch[tokens_torch >= self.size] = self.unk_token

        return tokens_torch

    def decode(self, tokens: torch.Tensor) -> str:
        return ''.join(chr(o.item()) for o in tokens)


class TikTokenTokenizer(Tokenizer):
    TYPE = 'tiktoken'

    def __init__(self, name: str = 'gpt2'):
        self.name = name
        self.encoding = tiktoken.get_encoding(name)

        real_size = self.encoding.n_vocab

        extra_pad = 0
        if not (real_size % 128 == 0):
            extra_pad = 128 - real_size % 128

        self.size_padded = extra_pad
    
    @property
    def size(self) -> int:
        return self.size_padded

    @property
    def eos_token_id(self) -> int:
        return self.encoding.eot_token

    def to_config(self):
        return {
            'name': self.name,
        }

    def encode(self, text: str, *, add_eos: bool = False) -> torch.Tensor:
        tokens = self.encoding.encode(text, disallowed_special=())
        if add_eos:
            tokens.append(self.eos_token_id)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.encoding.decode(tokens.tolist())


class HfTokenizer(Tokenizer):
    TYPE = 'hf'

    def __init__(self, name: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        real_size = self.tokenizer.vocab_size

        extra_pad = 0
        if not (real_size % 128 == 0):
            extra_pad = 128 - real_size % 128

        self.size_padded = real_size + extra_pad

    @classmethod
    def load(cls: Type[_T], path: str | Path, **extra_kwargs) -> _T:
        return cls(name=path)

    def save(self, path: str | Path) -> None:
        return self.tokenizer.save_pretrained(path)

    @property
    def size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode(self, text: str, *, add_eos: bool = False) -> torch.Tensor:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if add_eos:
            tokens.append(self.eos_token_id)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(tensor)

    def to_config(self):
        return {
            'name': self.name,
        }


if __name__ == '__main__':
    tokenizer = CharTokenizer()
    print(tokenizer.encode('I am a tokenizer!'))
