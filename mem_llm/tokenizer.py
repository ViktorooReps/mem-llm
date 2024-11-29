from abc import ABCMeta, abstractmethod

import tiktoken
import torch

from mem_llm.interface import Configurable


class Tokenizer(Configurable, metaclass=ABCMeta):
    TYPE = None

    @property
    def descriptor(self):
        return self.TYPE + '-' + '-'.join(f'{k}{v}' for k, v in self.to_config().items())

    @abstractmethod
    @property
    def size(self) -> int:
        pass

    @abstractmethod
    @property
    def eos_token(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        pass


class CharTokenizer(Tokenizer):
    TYPE = 'char'

    def __init__(self, vocab_size: int = 128, *, unk_token: int = 0, eos_token: int = 127):
        self._size = vocab_size

        self._unk_token = unk_token
        self._eos_token = eos_token

    @property
    def size(self) -> int:
        return self._size

    @property
    def unk_token(self) -> int:
        return self._unk_token

    @property
    def eos_token(self) -> int:
        return self._eos_token

    def to_config(self):
        return {
            'vocab_size': self.size,
            'unk_token': self.unk_token,
            'eos_token': self.eos_token
        }

    def encode(self, text: str, *, add_eos: bool = False) -> torch.Tensor:
        tokens = list(map(ord, text))
        if add_eos:
            tokens.append(self.eos_token)

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

    @property
    def size(self) -> int:
        return self.encoding.n_vocab

    @property
    def eos_token(self) -> int:
        return self.encoding.eot_token

    def to_config(self):
        return {
            'name': self.name,
        }

    def encode(self, text: str, *, add_eos: bool = False) -> torch.Tensor:
        tokens = self.encoding.encode(text)
        if add_eos:
            tokens.append(self.eos_token)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.encoding.decode(tokens.tolist())


TOKENIZERS = {
    CharTokenizer.TYPE: CharTokenizer,
    TikTokenTokenizer.TYPE: TikTokenTokenizer,
}


if __name__ == '__main__':
    tokenizer = CharTokenizer()
    print(tokenizer.encode('I am a tokenizer!'))
