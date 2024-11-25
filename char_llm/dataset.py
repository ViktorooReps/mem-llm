from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from threading import Lock
from typing import TypeVar, Callable

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from char_llm.custom_logging import logger
from char_llm.tokenizer import Tokenizer

from datasets import load_dataset as load_dataset_from_hub, DownloadConfig, DownloadMode

TS_DATASET_PATH = Path('tinyshakespeare.txt')
TS_PRETOKENIZED_DATASET_PATH = Path('data/tinyshakespeare')
TS_DATASET_SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

FWE_PRETOKENIZED_DATASET_PATH = Path('data/fineweb-edu')


class GuaranteedLengthDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, *, target_length: int):
        path = Path(path)
        assert path.exists() and path.is_dir()

        tokens_path = path / 'tokens.dat'
        assert tokens_path.exists() and tokens_path.is_file()

        self.target_length = target_length
        self.tokens = np.memmap(tokens_path, dtype=np.uint8, mode='r')

    def __len__(self):
        return len(self.tokens) - self.target_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        start_idx = idx
        end_idx = start_idx + self.target_length

        return torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.int)


def create_token_memmap(file_path: str | Path, shape: tuple[int, ...]):
    return np.memmap(file_path, dtype=np.uint8, mode='w+', shape=shape)


def save_memmap(file_path: str | Path, array: np.ndarray):
    file_path = Path(file_path)
    memmap_array = create_token_memmap(file_path, array.shape)
    memmap_array[:] = array[:]
    del memmap_array  # Flush changes and close memmap


@dataclass
class TSConfig:
    target_length: int
    split: float = 0.95

    dataset_name: str = 'tiny-shakespear'


def load_dataset_ts(config: TSConfig) -> (GuaranteedLengthDataset, GuaranteedLengthDataset):
    logger.info(f'Loading {config.dataset_name}')

    if TS_PRETOKENIZED_DATASET_PATH.exists():
        logger.info(f'Found {TS_PRETOKENIZED_DATASET_PATH}')
        return (
            GuaranteedLengthDataset(TS_PRETOKENIZED_DATASET_PATH / 'train', target_length=config.target_length),
            GuaranteedLengthDataset(TS_PRETOKENIZED_DATASET_PATH / 'val', target_length=config.target_length)
        )

    if not TS_DATASET_PATH.exists():
        response = requests.get(TS_DATASET_SOURCE)

        with open(TS_DATASET_PATH, 'wb') as file:
            file.write(response.content)

    source = TS_DATASET_PATH
    dest = TS_PRETOKENIZED_DATASET_PATH

    if not source.exists():
        raise FileNotFoundError

    dest.mkdir(parents=True, exist_ok=True)

    with open(source, 'r') as file:
        content = file.read()

    tokenizer = Tokenizer()
    tokenized = tokenizer.encode(content)

    np_tokens = tokenized.numpy().astype(np.uint8)

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

    save_memmap(train_path / 'tokens.dat', np_tokens_train)

    val_path = dest / 'val'
    val_path.mkdir(exist_ok=True)

    save_memmap(val_path / 'tokens.dat', np_tokens_val)

    return (
        GuaranteedLengthDataset(train_path, target_length=config.target_length),
        GuaranteedLengthDataset(val_path, target_length=config.target_length)
    )


@dataclass
class FineWebEduConfig:
    target_length: int
    train_length: int = 1_000_000_000
    val_length: int = 1_000_000

    download_config: DownloadConfig | None = None

    dataset_name: str = 'tiny-fineweb'


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
        lock: Lock
):
    """
    Tokenize an example and write it to memmap.
    Truncate tokens if they exceed the available space.
    """
    if isinstance(texts, str):
        texts = [texts]

    total_written = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_eos=True).numpy().astype(np.uint8)
        token_length = len(tokens)

        # Update the last location safely
        start_location, end_location = get_write_location(last_location, token_length, max_location, lock)
        to_write = end_location - start_location

        # Write the tokens on disk
        arr[start_location:end_location] = tokens[:to_write]

        total_written += to_write

    return total_written


def load_dataset_fwe(config: FineWebEduConfig) -> (GuaranteedLengthDataset, GuaranteedLengthDataset):
    logger.info(f"Loading {config.dataset_name}")

    if FWE_PRETOKENIZED_DATASET_PATH.exists():
        logger.info(f"Found {FWE_PRETOKENIZED_DATASET_PATH}")
        return (
            GuaranteedLengthDataset(FWE_PRETOKENIZED_DATASET_PATH / 'train', target_length=config.target_length),
            GuaranteedLengthDataset(FWE_PRETOKENIZED_DATASET_PATH / 'val', target_length=config.target_length),
        )

    dest = FWE_PRETOKENIZED_DATASET_PATH
    dest.mkdir(parents=True, exist_ok=True)

    # Paths for train and validation memmap files
    train_path = dest / "train"
    train_path.mkdir(exist_ok=True)
    train_file = train_path / "tokens.dat"

    val_path = dest / "val"
    val_path.mkdir(exist_ok=True)
    val_file = val_path / "tokens.dat"

    # Pre-allocate memmap files
    train_memmap = create_token_memmap(train_file, (config.train_length,))
    val_memmap = create_token_memmap(val_file, (config.val_length,))

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
    ), num_workers=4, prefetch_factor=8, batch_size=32)
    tokenizer = Tokenizer()

    # Iterate over the dataset and split into train/validation
    train_size = int(config.train_length)
    val_size = int(config.val_length)

    def process_example(example):
        total_written_train = train_last_location[0]
        total_written_val = val_last_location[0]

        if total_written_train < train_size:
            return tokenize_and_write(example, tokenizer, train_memmap, train_last_location, train_size, train_lock)
        elif total_written_val < val_size:
            return tokenize_and_write(example, tokenizer, val_memmap, val_last_location, val_size, val_lock)

        return 0

    with ThreadPoolExecutor() as executor:
        bar = tqdm(total=train_size + val_size)
        for batch in dataset_loader:
            futures = []

            for example in batch['text']:
                future = executor.submit(process_example, example)
                futures.append(future)

            stop_processing = False
            for future in futures:
                progress = future.result()
                bar.update(progress)

                if not progress:
                    stop_processing = True
                    break

            if stop_processing:
                break
        bar.close()

    logger.info(f"Total train tokens written: {train_last_location[0]}")
    logger.info(f"Total validation tokens written: {val_last_location[0]}")

    # Flush changes and close memmap
    del train_memmap
    del val_memmap

    return (
        GuaranteedLengthDataset(train_path, target_length=config.target_length),
        GuaranteedLengthDataset(val_path, target_length=config.target_length),
    )


DATASET_LOADERS = {
    TSConfig: load_dataset_ts,
    FineWebEduConfig: load_dataset_fwe
}


DATASET_CONFIGS = {config_type.dataset_name: config_type for config_type in DATASET_LOADERS}


def load_dataset(config) -> GuaranteedLengthDataset:
    return DATASET_LOADERS[type(config)](config)


_T = TypeVar('_T')


def register_dataset(config_type: _T, loader: Callable[[_T], GuaranteedLengthDataset]):
    assert hasattr(config_type, 'dataset_name')
    assert is_dataclass(config_type)
    DATASET_CONFIGS[config_type.dataset_name] = config_type
    DATASET_LOADERS[config_type] = loader


if __name__ == '__main__':
    tokenizer = Tokenizer()
    data_train, data_val = load_dataset_fwe(FineWebEduConfig(
        target_length=100,
        train_length=1000,
        val_length=1000
    ))

    logger.info(tokenizer.decode(data_train[500]))
    logger.info(tokenizer.decode(data_val[500]))