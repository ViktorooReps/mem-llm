from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from threading import Lock
from typing import TypeVar

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader

from mem_llm.custom_logging import logger
from mem_llm.custom_tqdm import HumanizedTqdm
from mem_llm.tokenizer import CharTokenizer, Tokenizer, TikTokenTokenizer

from datasets import load_dataset as load_dataset_from_hub, DownloadMode

DATA_PATH = Path('data')

TS_DATASET_PATH = DATA_PATH / 'tinyshakespeare.txt'
TS_DATASET_SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


@dataclass
class DatasetConfig:
    example_length: int
    tile_size: int | None = None

    # to be defined in subclasses
    dataset_name: str = None

    def get_cache_path(self, tokenizer: Tokenizer) -> Path:
        return DATA_PATH / f'{self.dataset_name}__{tokenizer.descriptor}'


def get_storage_type(tokenizer: Tokenizer) -> np.dtype:
    if isinstance(tokenizer, CharTokenizer):
        return np.uint8
    if isinstance(tokenizer, TikTokenTokenizer) and tokenizer.name == 'gpt2':
        return np.uint16  # around 50K tokens
    return np.uint32


class InfiniteDataLoaderWrapper:
    def __init__(self, dataloader: DataLoader):
        super().__init__()
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            yield from iter(self.dataloader)


class GuaranteedLengthDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path: str | Path,
            *,
            example_length: int,
            source_dtype: np.dtype,
            limit_dataset_length: int | None = None,
            tile_size: int | None = None
    ):
        path = Path(path)
        assert path.exists() and path.is_dir()

        tokens_path = path / 'tokens.dat'
        assert tokens_path.exists() and tokens_path.is_file()

        self.example_length = example_length
        self.tile_size = self.example_length // 2 if tile_size is None else tile_size
        self.tokens = np.memmap(tokens_path, dtype=source_dtype, mode='r')
        self.dataset_length = min(len(self.tokens), limit_dataset_length) if limit_dataset_length is not None else len(self.tokens)

    def __len__(self):
        return min((self.dataset_length - self.example_length) // self.tile_size, 1)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        idx = idx * self.tile_size

        start_idx = idx
        end_idx = start_idx + self.example_length

        return torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)


def create_token_memmap(file_path: str | Path, shape: tuple[int, ...], source_dtype: np.dtype):
    return np.memmap(file_path, dtype=source_dtype, mode='w+', shape=shape)


def save_memmap(file_path: str | Path, array: np.ndarray, source_dtype: np.dtype):
    file_path = Path(file_path)
    memmap_array = create_token_memmap(file_path, array.shape, source_dtype)
    memmap_array[:] = array[:]
    del memmap_array  # Flush changes and close memmap


@dataclass
class TSConfig(DatasetConfig):
    split: float = 0.95
    dataset_name: str = 'tiny-shakespeare'


def load_dataset_ts(config: TSConfig, tokenizer: Tokenizer) -> (GuaranteedLengthDataset, GuaranteedLengthDataset):
    logger.info(f"Loading {config.dataset_name} for {tokenizer.descriptor} tokenizer")

    source_dtype = get_storage_type(tokenizer)

    extras = {
        'source_dtype': source_dtype,
        'example_length': config.example_length,
        'tile_size': config.tile_size,
    }

    cache_path = config.get_cache_path(tokenizer)

    if cache_path.exists():
        logger.info(f'Found {cache_path}, loading the dataset from there')
        return (
            GuaranteedLengthDataset(cache_path / 'train', **extras, limit_dataset_length=None),
            GuaranteedLengthDataset(cache_path / 'val', **extras, limit_dataset_length=None)
        )

    if not TS_DATASET_PATH.exists():
        response = requests.get(TS_DATASET_SOURCE)

        TS_DATASET_PATH.parent.mkdir(exist_ok=True, parents=True)

        with open(TS_DATASET_PATH, 'wb') as file:
            file.write(response.content)

    source = TS_DATASET_PATH
    dest = cache_path

    if not source.exists():
        raise FileNotFoundError

    dest.mkdir(parents=True, exist_ok=True)

    with open(source, 'r') as file:
        content = file.read()

    tokenized = tokenizer.encode(content)

    np_tokens = tokenized.numpy().astype(source_dtype)

    n_token = len(np_tokens)

    logger.info(f'Read {n_token} tokens')

    n_train_tokens = int(len(np_tokens) * config.split)
    n_val_tokens = len(np_tokens) - n_train_tokens

    logger.info(f'Split: {n_train_tokens}/{n_val_tokens} tokens')

    train_token_mask = np.concatenate([np.ones(n_train_tokens, dtype=bool), np.zeros(n_val_tokens, dtype=bool)])
    val_token_mask = ~train_token_mask

    np_tokens_train = np_tokens[train_token_mask]
    np_tokens_val = np_tokens[val_token_mask]

    train_path = dest / 'train'
    train_path.mkdir(exist_ok=True)

    save_memmap(train_path / 'tokens.dat', np_tokens_train, source_dtype)

    val_path = dest / 'val'
    val_path.mkdir(exist_ok=True)

    save_memmap(val_path / 'tokens.dat', np_tokens_val, source_dtype)

    return (
        GuaranteedLengthDataset(train_path, **extras, limit_dataset_length=None),
        GuaranteedLengthDataset(val_path, **extras, limit_dataset_length=None)
    )


@dataclass
class FineWebEduConfig(DatasetConfig):
    num_download_workers: int = 12
    train_length: int = 20_000_000_000  # 20B
    val_length: int = 10_000_000  # 10M
    dataset_name: str = 'fineweb-edu'


def get_write_location(
        last_location: list[int],
        token_length: int,
        max_end_location: int,
        lock: Lock
) -> (int, int):

    with lock:
        start_location = last_location[0]
        last_location[0] += token_length
        last_location[0] = min(last_location[0], max_end_location)
        return start_location, last_location[0]  # [start, end)


def tokenize_and_write(
        texts: list[str] | str,
        tokenizer: Tokenizer,
        arr: np.memmap,
        last_location: list[int],
        max_location: int,
        lock: Lock,
        source_dtype: np.dtype
):
    """
    Tokenize an example and write it to memmap.
    Truncate tokens if they exceed the available space.
    """
    if isinstance(texts, str):
        texts = [texts]

    total_written = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_eos=True).numpy().astype(source_dtype)
        token_length = len(tokens)

        # Update the last location safely
        start_location, end_location = get_write_location(last_location, token_length, max_location, lock)
        to_write = end_location - start_location

        # Write the tokens on disk
        arr[start_location:end_location] = tokens[:to_write]

        total_written += to_write

    return total_written


def load_dataset_fwe(
        config: FineWebEduConfig,
        tokenizer: Tokenizer
) -> (GuaranteedLengthDataset, GuaranteedLengthDataset):

    logger.info(f"Loading {config.dataset_name} for {tokenizer.descriptor} tokenizer")

    source_dtype = get_storage_type(tokenizer)

    extras = {
        'source_dtype': source_dtype,
        'example_length': config.example_length,
        'tile_size': config.tile_size,
    }

    cache_path = config.get_cache_path(tokenizer)

    if cache_path.exists():
        logger.info(f"Found {cache_path}, loading the dataset from there")
        return (
            GuaranteedLengthDataset(cache_path / 'train', **extras, limit_dataset_length=config.train_length),
            GuaranteedLengthDataset(cache_path / 'val', **extras, limit_dataset_length=config.val_length),
        )

    dest = cache_path
    dest.mkdir(parents=True, exist_ok=True)

    # Paths for train and validation memmap files
    train_path = dest / "train"
    train_path.mkdir(exist_ok=True)
    train_file = train_path / "tokens.dat"

    val_path = dest / "val"
    val_path.mkdir(exist_ok=True)
    val_file = val_path / "tokens.dat"

    # Pre-allocate memmap files
    train_memmap = create_token_memmap(train_file, (config.train_length,), source_dtype)
    val_memmap = create_token_memmap(val_file, (config.val_length,), source_dtype)

    # Initialize shared indices and locks
    train_last_location = [0]  # Shared variable to track train memmap location
    val_last_location = [0]    # Shared variable to track val memmap location
    train_lock = Lock()
    val_lock = Lock()

    dataset_loader = DataLoader(load_dataset_from_hub(
        'HuggingFaceTB/smollm-corpus', 'fineweb-edu-dedup',
        streaming=True,
        split='train',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        columns=['text'],
        revision='3ba9d605774198c5868892d7a8deda78031a781f'
    ), num_workers=config.num_download_workers, prefetch_factor=8, batch_size=32)

    # Iterate over the dataset and split into train/validation
    train_size = int(config.train_length)
    val_size = int(config.val_length)

    def process_example(example) -> int | None:
        total_written_train = train_last_location[0]
        total_written_val = val_last_location[0]

        if total_written_train < train_size:
            return tokenize_and_write(
                example, tokenizer, train_memmap, train_last_location, train_size, train_lock, source_dtype
            )
        elif total_written_val < val_size:
            return tokenize_and_write(
                example, tokenizer, val_memmap, val_last_location, val_size, val_lock, source_dtype
            )

        return None

    with ThreadPoolExecutor(max_workers=config.num_download_workers) as executor:
        bar = HumanizedTqdm(total=train_size + val_size, unit='tokens')
        for batch in dataset_loader:
            futures = []

            for example in batch['text']:
                future = executor.submit(process_example, example)
                futures.append(future)

            stop_processing = False
            for future in futures:
                progress = future.result()
                if progress is None:
                    stop_processing = True
                    break

                bar.update(progress)

            if stop_processing:
                break
        bar.close()

    logger.info(f"Total train tokens written: {train_last_location[0]}")
    logger.info(f"Total validation tokens written: {val_last_location[0]}")

    # Flush changes and close memmap
    del train_memmap
    del val_memmap

    return (
        GuaranteedLengthDataset(train_path, **extras, limit_dataset_length=config.train_length),
        GuaranteedLengthDataset(val_path, **extras, limit_dataset_length=config.val_length),
    )


DATASET_LOADERS = {
    DatasetConfig: lambda *_, **__: (None, None),
    TSConfig: load_dataset_ts,
    FineWebEduConfig: load_dataset_fwe
}


DATASET_CONFIGS = {config_type.dataset_name: config_type for config_type in DATASET_LOADERS}


def load_dataset(config: DatasetConfig, tokenizer: Tokenizer) -> (GuaranteedLengthDataset, GuaranteedLengthDataset):
    """Returns train and validation splits"""
    return DATASET_LOADERS[type(config)](config, tokenizer)


_T = TypeVar('_T', bound=DatasetConfig)


def register_dataset(config_type: _T, loader):
    assert hasattr(config_type, 'dataset_name')
    assert is_dataclass(config_type)
    DATASET_CONFIGS[config_type.dataset_name] = config_type
    DATASET_LOADERS[config_type] = loader


if __name__ == '__main__':
    tok = CharTokenizer()
    data_train, data_val = load_dataset_fwe(FineWebEduConfig(
        example_length=100,
        train_length=1000,
        val_length=1000
    ), tok)

    logger.info(tok.decode(data_train[500]))
    logger.info(tok.decode(data_val[500]))
