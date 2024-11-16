import abc
import json
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
import torch
import re
import requests

from nltk.tokenize import PunktTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention.flex_attention import create_block_mask, and_masks, flex_attention
from torch.utils.data import DataLoader
from torch.nn.functional import scaled_dot_product_attention

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Iterable, TypeVar
from itertools import chain

from tqdm.auto import tqdm

nltk.download('punkt')
nltk.download('punkt_tab')


_T = TypeVar('_T')


DATASET_PATH = Path('tinyshakespeare.txt')
PRETOKENIZED_DATASET_PATH = Path('data/tinyshakespeare')
DATASET_SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
sdpa = torch.compile(scaled_dot_product_attention, dynamic=False)

torch._dynamo.config.cache_size_limit = 5000


class Configurable(metaclass=abc.ABCMeta):
    @abstractmethod
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
class ModelOutput:
    # model predicts next character and whether the current segment has ended
    logits: torch.Tensor
    logits_segment: torch.Tensor
    # per layer
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


class Generator(nn.Module, metaclass=abc.ABCMeta):
    @abstractmethod
    def forward(
            self,
            tokens: torch.Tensor,
            segment_mask: torch.Tensor,
            specials_mask: torch.Tensor,
            past_keys: list[torch.Tensor] | None = None,
            past_values: list[torch.Tensor] | None = None,
            **kwargs
    ) -> ModelOutput:
        pass


IS_SPECIAL_REGEX = re.compile(r'[^a-zA-Z0-9]')


class Tokenizer(Configurable):
    def __init__(
            self,
            vocab_size: int = 128,
            device: str = 'cpu',
            segment: str | None = None,  # "sentence", "word" or "sentence+word"
    ):
        self.size = vocab_size
        self.segment = segment if segment is not None else ''

        self.is_special = torch.tensor([
            IS_SPECIAL_REGEX.match(f'{chr(o)}') is not None
            for o in range(vocab_size)
        ], dtype=torch.bool, device=device)

        self.tokenizer_sent = None

        self.segment_sent = False
        self.segment_word = False

        self.unk_token = 0
        self.bot_token = 2
        self.eot_token = 3
        self.sep_token = ord(' ')
        self.pad_token = 127

        for segment_type in self.segment.split('+'):
            if segment_type == 'sentence':
                self.tokenizer_sent = PunktTokenizer('english')
                self.segment_sent = True
            if segment_type == 'word':
                self.segment_word = True

    @property
    def device(self):
        return self.is_special.device

    def to(self, device: str):
        self.is_special = self.is_special.to(device)
        return self

    def to_config(self):
        return {
            'size': self.size,
            'segment': self.segment,
        }

    def encode(self, text: str, *, add_sink: bool = True) -> dict:
        if self.segment_sent:
            segments = self.tokenizer_sent.span_tokenize(text)
        else:
            segments = [(0, len(text))]

        tokens = list(map(ord, text))

        token_segments = []
        segment_mask = []
        segment_starts = []

        prev_segm = None
        for segm in segments:
            start, end = segm

            if prev_segm is None:
                if add_sink:
                    # add sink
                    token_segments.append([self.bot_token])
                    segment_mask.append([False])
            elif prev_segm[1] == start:
                # separate the sentences
                token_segments.append([self.sep_token])
                segment_mask.append([False])
            else:
                prev_start, prev_end = prev_segm

                # everything between the sentences
                token_segments.append(tokens[prev_end:start])
                segment_mask.append([False] * (start - prev_end))

            token_segments.append(tokens[start:end])
            segment_mask.append([True] * (end - start))
            segment_starts.append(start)
            prev_segm = segm

        final_s, final_e = prev_segm
        if final_e != len(tokens):
            token_segments.append(tokens[final_e:len(tokens)])
            segment_mask.append([False] * (len(tokens) - final_e))

        all_tokens = chain.from_iterable(token_segments)
        all_segment_mask = chain.from_iterable(segment_mask)

        tokens_torch = torch.tensor(list(all_tokens), device=self.device, dtype=torch.int)
        tokens_torch[tokens_torch >= self.size] = self.unk_token

        return {
            'tokens': tokens_torch,
            'segment_mask': torch.tensor(list(all_segment_mask), device=self.device, dtype=torch.bool),
            'specials_mask': self.is_special[tokens_torch],
            'segment_starts': torch.tensor(segment_starts, device=self.device, dtype=torch.int),
        }

    def decode(self, tokens: Iterable[int]) -> str:
        # skip sink and pad tokens
        return ''.join(chr(o) for o in tokens if o != self.bot_token)

def generate_train_test_split(source: str | Path, dest: str | Path, *, split: float = 0.95):
    source = Path(source)
    dest = Path(dest)

    if not source.exists():
        raise FileNotFoundError

    dest.mkdir(parents=True, exist_ok=True)

    with open(source, 'r') as file:
        content = file.read()

    tokenizer = Tokenizer(segment='sentence')
    tokenized = tokenizer.encode(content, add_sink=False)

    np_tokens = tokenized['tokens'].numpy().astype(np.uint8)
    np_segment_mask = tokenized['segment_mask'].numpy().astype(bool)
    np_special_mask = tokenized['specials_mask'].numpy().astype(bool)
    np_segment_starts = tokenized['segment_starts'].numpy().astype(int)

    n_token  = len(np_tokens)
    n_sent = len(np_segment_starts)

    print(f'Read {n_token} tokens and {n_sent} sentences')

    n_sent = len(np_segment_starts)
    n_train_sent = int(n_sent * split)
    n_val_sent = n_sent - n_train_sent

    np_segment_ends = np.roll(np_segment_starts, -1)
    np_segment_ends[-1] = n_token
    segm_len = np_segment_ends - np_segment_starts

    n_train_tokens = int(segm_len[:n_train_sent].sum())
    n_val_tokens = len(np_tokens) - n_train_tokens

    print(f'Split: {n_train_tokens}/{n_val_tokens} tokens, {n_train_sent}/{n_val_sent} sentences')

    train_token_mask = torch.concat([
        torch.ones(n_train_tokens, dtype=torch.bool),
        torch.zeros(n_val_tokens, dtype=torch.bool)
    ])
    val_token_mask = ~train_token_mask

    train_sent_mask = torch.concat([
        torch.ones(n_train_sent, dtype=torch.bool),
        torch.zeros(n_val_sent, dtype=torch.bool)
    ])
    val_sent_mask = ~train_sent_mask

    np_tokens_train = np_tokens[train_token_mask]
    np_segment_mask_train = np_segment_mask[train_token_mask]
    np_special_mask_train = np_special_mask[train_token_mask]
    np_segment_starts_train = np_segment_starts[train_sent_mask]

    np_tokens_val = np_tokens[val_token_mask]
    np_segment_mask_val = np_segment_mask[val_token_mask]
    np_special_mask_val = np_special_mask[val_token_mask]
    np_segment_starts_val = np_segment_starts[val_sent_mask]

    train_path = dest / 'train'
    train_path.mkdir(exist_ok=True)

    np.save(train_path / 'tokens.npy', np_tokens_train)
    np.save(train_path / 'segment_mask.npy', np_segment_mask_train)
    np.save(train_path / 'special_mask.npy', np_special_mask_train)
    np.save(train_path / 'segment_starts.npy', np_segment_starts_train)

    val_path = path / 'val'
    val_path.mkdir(exist_ok=True)

    np.save(val_path / 'tokens.npy', np_tokens_val)
    np.save(val_path / 'segment_mask.npy', np_segment_mask_val)
    np.save(val_path / 'special_mask.npy', np_special_mask_val)
    np.save(val_path / 'segment_starts.npy', np_segment_starts_val)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, *, start_token: int, target_length: int):
        path = Path(path)
        assert path.exists() and path.is_dir()

        tokens_path = path / 'tokens.npy'
        segment_mask = path / 'segment_mask.npy'
        special_mask = path / 'special_mask.npy'
        segment_starts = path / 'segment_starts.npy'

        self.start_token = start_token
        self.target_length = target_length

        self.tokens = np.memmap(tokens_path, dtype=np.uint8, mode='r')
        self.segment_mask = np.memmap(segment_mask, dtype=bool, mode='r')
        self.special_mask = np.memmap(special_mask, dtype=bool, mode='r')
        self.segment_starts = np.load(segment_starts)

    def __len__(self):
        return len(self.segment_starts)

    def __getitem__(self, idx):
        start_idx = self.segment_starts[idx]
        end_idx = start_idx + self.target_length - 1  # reserve one token for start_token

        tokens = torch.tensor(self.tokens[start_idx:end_idx].astype(int))
        segment_mask = torch.tensor(self.segment_mask[start_idx:end_idx])
        special_mask = torch.tensor(self.special_mask[start_idx:end_idx])

        return {
            'tokens': torch.concatenate([tokens.new_tensor([self.start_token]), tokens]),
            'segment_mask': torch.concatenate([segment_mask.new_tensor([False]), segment_mask]),
            'specials_mask': torch.concatenate([special_mask.new_tensor([True]), special_mask])
        }


class Collator:
    def __init__(self, padding_token: int, move_to: str = 'cpu'):
        self.padding_token = padding_token
        self.move_to = move_to

    def __call__(self, batch):
        tokens = tuple(item['tokens'] for item in batch)
        tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=self.padding_token)

        segment_mask = tuple(item['segment_mask'] for item in batch)
        segment_mask_padded = pad_sequence(segment_mask, batch_first=True, padding_value=False)

        specials_mask = tuple(item['specials_mask'] for item in batch)
        specials_mask_padded = pad_sequence(specials_mask, batch_first=True, padding_value=False)

        return {
            'tokens': tokens_padded.to(self.move_to),
            'segment_mask': segment_mask_padded.to(self.move_to),
            'specials_mask': specials_mask_padded.to(self.move_to),
        }


def calculate_intervals(boundary_mask):
    # Find cumulative sum of the gaps to identify different intervals
    interval_idx = torch.cumsum(boundary_mask, dim=-1) + 1
    shifted = interval_idx.roll(1)
    shifted[:, 0] = interval_idx[:, 0]

    return shifted


def create_mask(
        tokens: torch.Tensor,
        segment_mask: torch.Tensor,
        specials_mask: torch.Tensor,
        *,
        padding_token: int,
        cache_size: int = 0,
        mask_words_distance: int = 0,
        mask_segments_distance: int = 0,
        device: str | None = None
):
    if device is None:
        device = tokens.device

    batch_size, seq_length = tokens.shape

    word_interval_idx = calculate_intervals(specials_mask)

    segment_boundaries_mask = ~segment_mask
    segment_interval_idx = calculate_intervals(segment_boundaries_mask)

    def causal_mask(b, h, q_idx, kv_idx):
        return (q_idx + cache_size) >= kv_idx

    def padding_mask(b, h, q_idx, kv_idx):
        q_token = tokens[b, q_idx + cache_size]
        kv_token = tokens[b, kv_idx]
        return torch.ne(q_token, padding_token) & torch.ne(kv_token, padding_token)

    def word_mask(b, h, q_idx, kv_idx):
        q_interval_id = word_interval_idx[b, q_idx + cache_size]
        kv_interval_id = word_interval_idx[b, kv_idx]

        return (torch.eq(specials_mask[b, kv_idx], True)
                | ((q_idx + cache_size) - kv_idx < mask_words_distance)
                | torch.eq(q_interval_id, kv_interval_id))

    def segment_mask(b, h, q_idx, kv_idx):
        q_interval_id = segment_interval_idx[b, q_idx + cache_size]
        kv_interval_id = segment_interval_idx[b, kv_idx]

        return (torch.eq(segment_boundaries_mask[b, kv_idx], True)
                | ((q_idx + cache_size) - kv_idx < mask_segments_distance)
                | torch.eq(q_interval_id, kv_interval_id))

    return create_block_mask(
        mask_mod=and_masks(causal_mask, padding_mask, word_mask, segment_mask),
        B=batch_size,
        H=None,
        Q_LEN=seq_length - cache_size,
        KV_LEN=seq_length,
        device=device,
        _compile=True,
        BLOCK_SIZE=128,
    )


@torch.no_grad()
def generate(
        seed: str,
        model: Generator,
        tokenizer: Tokenizer,
        *,
        device: str,
        max_length: int,
        amp_enabled: bool = True,
        progress_bar: bool = True,
) -> str:
    model.eval()

    model = model.to(device)
    tokenizer = tokenizer.to(device)

    inputs = tokenizer.encode(seed)
    tokens = inputs['tokens']
    segment_mask = inputs['segment_mask']
    specials_mask = inputs['specials_mask']

    past_keys = None
    past_values = None

    for _ in tqdm(
            range(len(tokens), max_length),
            total=max_length - len(tokens),
            desc="Generating",
            disable=not progress_bar
    ):
        with torch.amp.autocast(enabled=amp_enabled, device_type=device):
            outputs = model(
                # batch_size = 1
                tokens=tokens.unsqueeze(0),
                segment_mask=segment_mask.unsqueeze(0),
                specials_mask=specials_mask.unsqueeze(0),
                past_keys=past_keys,
                past_values=past_values
            )

        # update current state of masks and tokens

        predicted_token = torch.argmax(outputs.logits[0, -1])
        predicted_segment = torch.argmax(outputs.logits_segment[0, -1]).to(dtype=torch.bool)
        is_predicted_special = tokenizer.is_special[predicted_token.item()]

        tokens = torch.concatenate([tokens, predicted_token.view(1)], dim=0)
        segment_mask = torch.concatenate([segment_mask, predicted_segment.view(1)], dim=0)
        specials_mask = torch.concatenate([specials_mask, is_predicted_special.view(1)], dim=0)

        # update past keys and values

        past_keys = outputs.keys
        past_values = outputs.values

    return tokenizer.decode(tokens)


class DummyModel(Generator):
    def __init__(
            self,
            vocab_size: int = 128,
            n_layers: int = 36,
            head_dim: int = 128,
            n_heads: int = 64,
            pad_token: int = 127,
            device: str = 'cpu',
            attn_impl: str = 'flex',
    ):
        nn.Module.__init__(self)

        self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.hidden_dim = self.n_heads * self.head_dim
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.attn_impl = attn_impl

        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim, device=device, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, device=device, dtype=torch.bfloat16)
        self.segment_head = nn.Linear(self.hidden_dim, 2, device=device, dtype=torch.bfloat16)

    def forward(
            self,
            tokens: torch.Tensor,
            segment_mask: torch.Tensor,
            specials_mask: torch.Tensor,
            past_keys: list[torch.Tensor] | None = None,
            past_values: list[torch.Tensor] | None = None,
            **_,
    ):
        batch_size, seq_length = tokens.shape
        device = tokens.device
        assert device == self.emb.weight.device

        if past_keys is None or not len(past_keys):
            past_keys = [
                torch.empty((batch_size, 0, self.hidden_dim), device=device, dtype=torch.bfloat16)
                for _ in range(self.n_layers)
            ]

        if past_values is None or not len(past_values):
            past_values = [
                torch.empty((batch_size, 0, self.hidden_dim), device=device, dtype=torch.bfloat16)
                for _ in range(self.n_layers)
            ]

        cache_size = past_keys[0].shape[1]
        # cache at each layer should be of equal size
        assert all(cache.shape[1] == cache_size for cache in past_values)
        assert all(cache.shape[1] == cache_size for cache in past_keys)

        new_len = seq_length - cache_size

        hidden = self.emb(tokens[:, cache_size:])

        if self.attn_impl == 'flex':
            mask = create_mask(
                tokens, segment_mask, specials_mask,
                cache_size=cache_size,
                padding_token=self.pad_token
            )

        for layer_idx in range(self.n_layers):
            # read cache
            hidden_k = torch.concatenate([hidden, past_keys[layer_idx]], dim=1)
            hidden_v = torch.concatenate([hidden, past_values[layer_idx]], dim=1)

            # update cache
            past_keys[layer_idx] = hidden_k.detach()
            past_values[layer_idx] = hidden_v.detach()

            # calculate new hidden for next layer

            # (B, S, H) -> (B, kh, S, Hh)
            hidden = hidden.view(batch_size, new_len, self.n_heads, self.head_dim).transpose(1, 2)
            hidden_k = hidden_k.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
            hidden_v = hidden_v.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

            if self.attn_impl == 'flex':
                hidden = flex_attention(hidden, hidden_k, hidden_v, block_mask=mask)
            elif self.attn_impl == 'sdpa':
                hidden = sdpa(hidden, hidden_k, hidden_v, is_causal=True)
            hidden = hidden.transpose(1, 2).contiguous().view(batch_size, new_len, -1)

        return ModelOutput(
            # project from hidden space to logit space
            logits=self.lm_head(hidden),
            logits_segment=self.segment_head(hidden),
            keys=past_keys,
            values=past_values
        )


if __name__ == '__main__':
    print(f'Device detected: {torch.cuda.get_device_name(0)}')

    to_encode = 'I am a token. You are a token...........  So what?'
    print(f'Encoding example for {to_encode}:')
    print(Tokenizer(segment='sentence+word').encode(to_encode))

    if not DATASET_PATH.exists():
        print(f'Downloading {DATASET_SOURCE}...')
        response = requests.get(DATASET_SOURCE)

        with open(DATASET_PATH, 'wb') as file:
            file.write(response.content)

        print(f'Saved {DATASET_SOURCE} at {DATASET_PATH}')
    else:
        print(f'Found dataset at {DATASET_PATH}')

    if not PREPROCESSED_PATH.exists():
        print('Generating train/val splits...')
        generate_train_test_split(DATASET_SOURCE, PRETOKENIZED_DATASET_PATH)
    else:
        print(f'Found preprocessed dataset at {PRETOKENIZED_DATASET_PATH}')

    impl = 'flex'

    model = torch.compile(DummyModel(device='cuda', attn_impl='flex'), dynamic=True)
    tokenizer = Tokenizer(device='cuda', segment='word+sentence')

    print('Testing model...')
    print('Generation: ', end='')

    res = generate(test_str, model, tokenizer, max_length=len(test_str) + 100, device='cuda', amp_enabled=True)
    if res is not None:
        print('Ok!')
    else:
        print('Failed!')


