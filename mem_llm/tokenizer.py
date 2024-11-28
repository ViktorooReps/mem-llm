import torch

from mem_llm.interface import Configurable


class Tokenizer(Configurable):
    def __init__(self, vocab_size: int = 128, *, unk_token: int = 0, eos_token: int = 127):
        self.size = vocab_size

        self.unk_token = unk_token
        self.eos_token = eos_token

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

        tokens_torch = torch.tensor(tokens, dtype=torch.int)
        tokens_torch[tokens_torch >= self.size] = self.unk_token

        return tokens_torch

    def decode(self, tokens: torch.Tensor) -> str:
        return ''.join(chr(o.item()) for o in tokens)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    print(tokenizer.encode('I am a tokenizer!'))
