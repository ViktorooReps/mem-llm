import argparse
import dataclasses
import os.path
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TypeVar, Type

import torch
import random
import numpy as np
from muon import Muon
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from mem_llm.custom_logging import logger
from mem_llm.custom_tqdm import abbreviate_number
from mem_llm.dataset import DATASET_CONFIGS, load_dataset, InfiniteDataLoaderWrapper
from mem_llm.interface import Configurable, ModelOutput
from mem_llm.model import MemLLM
from mem_llm.tokenizer import TOKENIZERS

RUNS_DIR = 'runs'
CPTS_DIR = 'checkpoints'
MODEL_DIR = 'model'

METRICS_FILE = 'metrics.csv'
CONFIG_FILE = 'config.json'
OPTIMIZER_FILE = 'optimizer.pt'
SCHEDULER_FILE = 'scheduler.pt'
TRAINING_STATE_FILE = 'training_state.pt'


@dataclass
class TrainingConfig(Configurable):
    # training length
    training_steps: int = 100_000
    cooldown_steps: int | float = 0.2
    warmup_steps: int | float = 1000

    # evaluation
    eval_steps: int = 50
    eval_per_steps: int = 200

    # checkpointing
    checkpoint_per_steps: int = 10_000
    checkpoints_dir: str = None  # determined automatically with run_name

    # optimizer
    use_muon: bool = True
    muon_momentum: float = 0.95
    muon_learning_rate: float = 0.02
    adamw_learning_rate: float = 0.01
    adamw_beta1: float = 0.8
    adamw_beta2: float = 0.95
    adamw_wd: float = 0.01

    # model
    dataset_config: dict = field(default_factory=lambda: {'dataset_name': 'char-fineweb-edu'})
    model_config: dict = field(default_factory=dict)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    do_compile: bool = True

    # tokenizer
    tokenizer_type: str = 'tiktoken'
    tokenizer_config: dict = field(default_factory=dict)

    # logging
    run_name: str = None
    run_dir: str = None  # determined automatically with run_name
    metrics_file: str = None  # determined automatically with run_name
    log_per_steps: int = 50
    disable_progress_bar: bool = False
    rewrite_metrics: bool = True

    # some extras
    num_dataloader_workers: int = 4
    dataloader_prefetch_factor: int = 8

    @classmethod
    def for_run(cls, run_name: str) -> 'TrainingConfig':
        return cls.load(os.path.join(RUNS_DIR, run_name))

    def to_config(self):
        return dataclasses.asdict(self)

    def __post_init__(self):
        if self.run_name is None:
            raise ValueError('run_name cannot be None')

        if self.run_dir is None:
            self.run_dir = os.path.join(RUNS_DIR, self.run_name)

        if self.metrics_file is None:
            self.metrics_file = os.path.join(self.run_dir, METRICS_FILE)

        if self.checkpoints_dir is None:
            self.checkpoints_dir = os.path.join(self.run_dir, CPTS_DIR)

        if isinstance(self.cooldown_steps, float):
            self.cooldown_steps = int(self.cooldown_steps * self.training_steps)

        if isinstance(self.warmup_steps, float):
            self.warmup_steps = int(self.warmup_steps * self.training_steps)

        assert self.eval_steps > 0
        assert self.training_steps >= self.warmup_steps + self.cooldown_steps


class MetricsLogger(Configurable):
    def __init__(self, log_file: str | Path, column_names: list[str], *, rewrite: bool = True):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True, parents=True)

        if self.log_file.suffix != '.csv':
            self.log_file = self.log_file.with_suffix('.csv')

        self.column_names = list(column_names)

        # Initialize the log file with column headers
        if not self.log_file.exists() or rewrite:
            with open(self.log_file, 'w') as f:
                f.write(','.join(self.column_names) + '\n')

    def log(self, metrics: dict):
        # Ensure no new keys are introduced
        extra_keys = [key for key in metrics if key not in self.column_names]
        if extra_keys:
            logger.error(
                f'Unexpected metrics: {extra_keys}. Only the following metrics are allowed: {self.column_names}'
            )

        complete_metrics = {key: metrics.get(key, None) for key in self.column_names}

        with open(self.log_file, 'a') as f:
            # Write the values in the same order as the column names
            row = [str(complete_metrics[key]) if complete_metrics[key] is not None else '' for key in self.column_names]
            f.write(','.join(row) + '\n')

    def to_config(self):
        return {
            'log_file': str(self.log_file),
            'column_names': self.column_names,
        }


_T = TypeVar('_T')


@dataclass
class TrainingContext:
    # stored in checkpoint
    config: TrainingConfig
    model: MemLLM
    optimizer: Optimizer
    scheduler: LRScheduler
    step: int

    # re-instantiated
    metrics_logger: MetricsLogger
    train_dataloader: InfiniteDataLoaderWrapper
    eval_dataloader: InfiniteDataLoaderWrapper

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert path.is_dir()

        config_path = path / CONFIG_FILE
        model_state_path = path / MODEL_DIR
        optimizer_state_path = path / OPTIMIZER_FILE
        scheduler_state_path = path / SCHEDULER_FILE
        training_state_path = path / TRAINING_STATE_FILE

        self.config.save(config_path)
        self.model.save(model_state_path)
        torch.save(self.optimizer.state_dict(), optimizer_state_path)
        torch.save(self.scheduler.state_dict(), scheduler_state_path)

        training_state = {
            'step': self.step,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            },
        }
        torch.save(training_state, training_state_path)

    @classmethod
    def load(cls: Type[_T], path: str | Path) -> _T:
        """Load the training state from a checkpoint file.

        # FIXME: a bit of a hack, but there is really no need to load multiple contexts
        WARNING: this sets RNG seeds, you are not supposed to instantiate multiple contexts in one process!
        """
        path = Path(path)
        assert path.exists() and path.is_dir()

        config = TrainingConfig.load(path / CONFIG_FILE)
        config.rewrite_metrics = False

        context = new_context(config, pretrained_model_path=path / MODEL_DIR)
        context.optimizer.load_state_dict(torch.load(path / OPTIMIZER_FILE, map_location='cpu'))
        context.scheduler.load_state_dict(torch.load(path / SCHEDULER_FILE, map_location='cpu'))

        training_state = torch.load(path / TRAINING_STATE_FILE, weights_only=False)

        context.step = training_state['step']

        rng_state = training_state['rng_state']
        torch.set_rng_state(rng_state['torch'])
        torch.cuda.set_rng_state_all(rng_state['cuda'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['python'])

        return context


def trapezoid_schedule(step: int, *, total_steps: int, warmup_steps: int, cooldown_steps: int) -> float:
    cooldown_start = total_steps - cooldown_steps

    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    # 2) constant lr for a while
    elif step < cooldown_start:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (total_steps - step) / cooldown_steps
        return decay_ratio


def new_context(config: TrainingConfig, *, pretrained_model_path: str | Path | None = None) -> TrainingContext:
    """Instantiates a new training context with the model initialized to pretrained_model_path if the latter is
    not None"""

    metrics_logger = MetricsLogger(config.metrics_file, column_names=[
        'step', 'seen_tokens', 'time_delta_s', 'run_name', 'lr_mult', 'train_loss', 'val_loss',
    ], rewrite=config.rewrite_metrics)

    tokenizer = TOKENIZERS[config.tokenizer_type].from_config(config.tokenizer_config)

    dataset_name = config.dataset_config['dataset_name']
    dataset_config = DATASET_CONFIGS[dataset_name](**config.dataset_config)
    train_data, eval_data = load_dataset(dataset_config, tokenizer)

    # we set batch_size to None because the batch size is dictated by the example length
    train_dataloader = InfiniteDataLoaderWrapper(DataLoader(
        train_data, None,
        num_workers=config.num_dataloader_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
    ))
    eval_dataloader = InfiniteDataLoaderWrapper(DataLoader(
        eval_data, None,
        num_workers=config.num_dataloader_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
    ))

    if pretrained_model_path is not None:
        model = MemLLM.load(pretrained_model_path, device=config.device)
    else:
        model = MemLLM.from_config(config.model_config, device=config.device)

    if config.use_muon:
        # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
        muon_params = [p for p in model.transformer_blocks.parameters() if p.ndim >= 2]

        # Find everything else -- these will be optimized by AdamW
        adamw_params = [p for p in model.parameters() if p.ndim < 2]
        adamw_params.extend(model.lm_head.parameters())
        adamw_params.extend(model.token_embedding.parameters())

        optimizer = Muon(
            muon_params=muon_params,
            lr=config.muon_learning_rate,
            momentum=config.muon_momentum,
            adamw_params=adamw_params,
            adamw_lr=config.adamw_learning_rate,
            adamw_betas=(config.adamw_beta1, config.adamw_beta2),
            adamw_wd=config.adamw_wd,
        )
    else:
        optimizer = AdamW(
            params=model.parameters(),
            lr=config.adamw_learning_rate,
            betas=(config.adamw_beta1, config.adamw_beta2),
            weight_decay=config.adamw_wd,
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=partial(
            trapezoid_schedule,
            total_steps=config.training_steps,
            warmup_steps=config.warmup_steps,
            cooldown_steps=config.cooldown_steps
        )
    )

    return TrainingContext(
        step=0,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics_logger=metrics_logger,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )


def list_checkpoints(config: TrainingConfig) -> list[Path]:
    checkpoint_dir = Path(config.checkpoints_dir)

    if checkpoint_dir.exists():
        return list(filter(lambda f: f.is_dir(), checkpoint_dir.glob('checkpoint_*')))

    return []


def load_checkpoint(checkpoint_dir: str | Path) -> TrainingContext:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f'You are trying to load from checkpoint directory that does not exist: {checkpoint_dir}'
        )

    return TrainingContext.load(checkpoint_dir)


def save_checkpoint(context: TrainingContext):
    checkpoints_dir = Path(context.config.checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    assert checkpoints_dir.is_dir()

    save_to_checkpoint_dir = checkpoints_dir / f'checkpoint_{context.step}'
    context.save(save_to_checkpoint_dir)


def prepare_context(config: TrainingConfig) -> TrainingContext:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.is_dir():
        raise NotADirectoryError(f'run_dir {run_dir} is not a directory')

    config_path = run_dir / CONFIG_FILE
    if config_path.exists():
        old_config = TrainingConfig.load(config_path)
        if old_config != config:
            raise RuntimeError(f'Config at {config_path} does not match current config! '
                               f'Have you forgotten to change run_name?')
        
        # check if the experiment has already been completed
        model_path = run_dir / MODEL_DIR
        if model_path.exists() and model_path.is_dir() and len(list(model_path.iterdir())):
            raise RuntimeError(f'The experiment at {run_dir} has already been finished! See trained model at {model_path}')

        # try to load the last checkpoint
        checkpoints = list_checkpoints(config)
        if len(checkpoints):
            last_checkpoint = checkpoints[-1]
            logger.warning(f'Continuing {config.run_name} from the last checkpoint: {last_checkpoint}')
            config.rewrite_metrics = False  # continue the run, so keep old metrics
            return load_checkpoint(last_checkpoint)
        
        metrics_file = run_dir / METRICS_FILE
        if metrics_file.exists():
            logger.critical(f'Rewriting experiment data at {run_dir}! You have 10 seconds to stop this')
            time.sleep(10.0)
            config.rewrite_metrics = True
    else:
        config.save(config_path)

    # fresh run
    return new_context(config)


@torch.no_grad()
def evaluate(context: TrainingContext):
    context.model.eval()

    config = context.config

    # this is fine since our samplers are infinite
    eval_dataloader = iter(context.eval_dataloader)

    running_eval_loss_sum = 0.0

    eval_start_time = time.time()

    for _ in tqdm(
            range(config.eval_steps),
            desc='Evaluating',
            total=config.eval_steps,
            disable=config.disable_progress_bar,
            leave=False
    ):
        example = next(eval_dataloader)

        target = torch.roll(example, shifts=-1)

        # there is no target for the last token
        example = example[:-1].to(config.device)
        target = target[:-1].to(config.device)

        seq_length = len(example)

        outputs: ModelOutput = context.model(example)  # noqa
        running_eval_loss_sum += F.cross_entropy(outputs.logits.view(seq_length, -1), target.view(seq_length)).item()

    if config.device.startswith('cuda'):
        # to compute time delta correctly, should not impact the performance too much
        torch.cuda.synchronize()

    context.metrics_logger.log({
        'step': context.step,
        'run_name': config.run_name,
        'seen_tokens': context.step * len(example),  # we guarantee the example length with our dataset
        'time_delta_s': time.time() - eval_start_time,
        'lr_mult': trapezoid_schedule(
            context.step,
            total_steps=config.training_steps,
            warmup_steps=config.warmup_steps,
            cooldown_steps=config.cooldown_steps
        ),
        'val_loss': running_eval_loss_sum / config.eval_steps,
    })


def train(context: TrainingContext):
    context.model.train()

    model_params = sum(p.numel() for p in context.model.parameters())
    logger.info(f'Running training for model with {abbreviate_number(model_params)} parameters')

    config = context.config

    # this will make a first few steps extremely slow
    context.model = torch.compile(context.model, disable=not config.do_compile)

    last_step = context.step

    if last_step:
        logger.warning(f'Continuing training from last checkpoint at {last_step} step')

    # this is fine since our samplers are infinite
    train_dataloader = iter(context.train_dataloader)

    running_train_loss_sum = 0.0
    steps_without_log = 0

    last_log_time = time.time()

    bar = tqdm(desc='Training', total=config.training_steps, disable=config.disable_progress_bar, leave=True)

    for step_idx in range(last_step, config.training_steps):
        context.step = step_idx
        bar.update(context.step - bar.n)

        example = next(train_dataloader)

        target = torch.roll(example, shifts=-1)

        # there is no target for the last token
        example = example[:-1].to(config.device)
        target = target[:-1].to(config.device)

        seq_length = len(example)

        def closure():
            context.optimizer.zero_grad()

            outputs: ModelOutput = context.model(example)  # noqa
            loss = F.cross_entropy(outputs.logits.view(seq_length, -1), target.view(seq_length))

            loss.backward()

            return loss.item()

        running_train_loss_sum += context.optimizer.step(closure)
        context.scheduler.step()

        steps_without_log += 1

        if step_idx > 0 and step_idx % config.checkpoint_per_steps == 0:
            save_checkpoint(context)

        if step_idx > 0 and step_idx % config.eval_per_steps == 0:
            evaluate(context)

        if step_idx > 0 and step_idx % config.log_per_steps == 0:
            running_train_loss = running_train_loss_sum / steps_without_log

            running_train_loss_sum = 0.0
            steps_without_log = 0

            if config.device.startswith('cuda'):
                # to compute time delta correctly, should not impact the performance too much,
                # given log_per_steps are reasonable
                torch.cuda.synchronize()

            context.metrics_logger.log({
                'step': context.step,
                'run_name': config.run_name,
                'seen_tokens': context.step * len(example),  # we guarantee the example length with our dataset
                'time_delta_s': time.time() - last_log_time,
                'lr_mult': trapezoid_schedule(
                    context.step,
                    total_steps=config.training_steps,
                    warmup_steps=config.warmup_steps,
                    cooldown_steps=config.cooldown_steps
                ),
                'train_loss': running_train_loss,
            })

            last_log_time = time.time()

    save_path = Path(config.run_dir) / MODEL_DIR
    logger.info(f'Saving trained model to {save_path}')

    context.model.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training from configuration.')

    parser.add_argument(
        'run_name',
        type=str,
        help=f'Name of the run (required). The config file is expected to be at '
             f'{os.path.join(RUNS_DIR, "<run_name>", CONFIG_FILE)}'
    )

    parser.add_argument(  # TODO: work in cooled/at<step>/ directory. tinker with the config
        '--cooldown_checkpoints',
        type=int,
        nargs='+',
        default=[],
        help='One or more integers specifying checkpoint save steps.'
    )

    parser.add_argument(
        '--from_config',
        type=str,
        default=None,
        help='Use specified config location instead of default.'
    )

    args = parser.parse_args()
    if len(args.cooldown_checkpoints):
        raise NotImplementedError()

    if args.from_config is not None:
        cfg_path = Path(args.from_config)
    else:
        cfg_path = Path(os.path.join(RUNS_DIR, args.run_name, CONFIG_FILE))

    if not cfg_path.exists():
        raise FileNotFoundError(f'Could not locate a config file at {cfg_path}')

    cfg = TrainingConfig.load(cfg_path)
    if cfg.run_name is None:
        cfg.run_name = args.run_name

    ctx = prepare_context(cfg)
    train(ctx)
