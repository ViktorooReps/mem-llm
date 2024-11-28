import pandas as pd
import torch

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from tqdm.auto import tqdm

from typing import Iterator

from torch.nn.functional import scaled_dot_product_attention



def create_mask_mod(context_size: int, stm_window_size: int, mem_freq: int):
    """
    Designed for the use on the input that consists of the concatenation of memory states and context states.
    Does not support padding!
    """
    n_mem = (context_size // mem_freq) + (context_size % mem_freq > 0)  # for block_size = 4: 0, 4, 8, 12, ...
    main_start = n_mem

    mem_end = main_start

    def causal_window_mask_with_mem(b, h, q_idx, kv_idx):
        # differentiator of main from mem parts
        is_mem_kv = (kv_idx < mem_end)
        is_mem_q = (q_idx < mem_end)

        # the first tokens are mem, so we realign with 0
        main_kv_idx = kv_idx - main_start
        main_q_idx = q_idx - main_start

        # mem tokens are really at every mem_freq position
        mem_kv_idx = kv_idx * mem_freq
        mem_q_idx = q_idx * mem_freq

        causal_main_diagonal = (kv_idx <= q_idx)
        main2main_windowed = (main_q_idx - main_kv_idx < stm_window_size)  # [s, e)

        # without window when attending to mem
        # <= to include the diagonal
        main2mem_causal = (mem_kv_idx <= main_q_idx)

        # do not include the diagonal!
        mem2main_causal = (main_kv_idx < mem_q_idx)
        # we did not include the diagonal, so add 1 token more here
        mem2main_windowed = (mem_q_idx - main_kv_idx <= stm_window_size)  # (s, e]

        case_main2main = (~is_mem_kv & ~is_mem_q & causal_main_diagonal & main2main_windowed)
        case_main2mem = (is_mem_kv & ~is_mem_q & main2mem_causal)
        case_mem2mem = (is_mem_kv & is_mem_q & causal_main_diagonal)
        case_mem2main = (~is_mem_kv & is_mem_q & mem2main_causal & mem2main_windowed)
        return case_main2main | case_main2mem | case_mem2mem | case_mem2main  # (3)
    return causal_window_mask_with_mem


# benchmark forward
batch_size = 1

block_size = 128
window_size = block_size * 8
compile_warmup = 25
n_trials = 100

min_blocks = 1
max_blocks = 1000

# llama3 8b
kv_heads = 8
q_heads = 32
head_dim = 4096 // q_heads

data = []
context_sizes = [block_size * i for i in range(min_blocks, max_blocks, 50)]

mem_freqs = [2, 8, 32, 128]


def benchmark(
        attn_impl, input_len: int, res_base: dict, do_compile: bool = True, dtype=torch.float16,
        **extra_kwargs
) -> Iterator[dict]:
    torch.compiler.reset()
    torch.cuda.empty_cache()
    attn_impl = torch.compile(attn_impl) if do_compile else attn_impl

    # compilation warmup
    if do_compile:
        for _ in range(compile_warmup):
            torch.cuda.empty_cache()

            input_k = torch.randn(size=(batch_size, kv_heads, input_len, head_dim), dtype=dtype, device='cuda').contiguous()
            input_v = torch.randn(size=(batch_size, kv_heads, input_len, head_dim), dtype=dtype, device='cuda').contiguous()
            input_q = torch.randn(size=(batch_size, q_heads, input_len, head_dim), dtype=dtype, device='cuda').contiguous()

            with torch.no_grad():
                assert attn_impl(input_q, input_k, input_v, **extra_kwargs).sum() is not None
        torch.cuda.synchronize(device=None)

    for _ in range(n_trials):
        torch.cuda.empty_cache()

        input_k = torch.randn(size=(batch_size, kv_heads, input_len, head_dim), dtype=dtype, device='cuda').contiguous()
        input_v = torch.randn(size=(batch_size, kv_heads, input_len, head_dim), dtype=dtype, device='cuda').contiguous()
        input_q = torch.randn(size=(batch_size, q_heads, input_len, head_dim), dtype=dtype, device='cuda').contiguous()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(device=None)

        with torch.autograd.profiler.profile(use_device='cuda') as prof:
            with torch.no_grad():
                assert attn_impl(input_q, input_k, input_v, **extra_kwargs).sum() is not None
                torch.cuda.synchronize(device=None)

        peak_mem = torch.cuda.max_memory_allocated()

        time_ns = prof.profiling_end_time_ns - prof.profiling_start_time_ns

        # Record results
        yield {
            **res_base,
            'time_taken_ms': time_ns / 1e6,
            'memory_used_mb': peak_mem / 1024 / 1024
        }


# causal sdpa

for context_size in tqdm(context_sizes):
    try:
        for res in benchmark(
                scaled_dot_product_attention, context_size,
                {'impl': 'sdpa_causal', 'context_length': context_size},
                enable_gqa=True, is_causal=True
        ):
            data.append(res)
    except Exception as e:
        print('Exc!')
        print(str(e), e.__class__.__name__)


# causal flex

def causal_mask_mod(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


for context_size in tqdm(context_sizes):
    mask = create_block_mask(causal_mask_mod, None, None, context_size, context_size, device='cuda',
                             BLOCK_SIZE=block_size, _compile=True)

    try:
        for res in benchmark(
                flex_attention, context_size,
                {'impl': 'flex_causal', 'context_length': context_size},
                enable_gqa=True, block_mask=mask
        ):
            data.append(res)
    except Exception as e:
        print('Exc!')
        print(str(e), e.__class__.__name__)


# causal window attention

def window_mask_mod(b, h, q_idx, kv_idx):
    return (kv_idx <= q_idx) & (q_idx - kv_idx < window_size)


for context_size in tqdm(context_sizes):
    mask = create_block_mask(window_mask_mod, None, None, context_size, context_size, device='cuda',
                             BLOCK_SIZE=block_size, _compile=True)

    try:
        for res in benchmark(
                flex_attention, context_size,
                {'impl': 'flex_window', 'context_length': context_size},
                enable_gqa=True, block_mask=mask
        ):
            data.append(res)
    except Exception as e:
        print('Exc!')
        print(str(e), e.__class__.__name__)



# causal window attention with memory

for context_size in tqdm(context_sizes):
    try:
        for mem_freq in tqdm(mem_freqs, leave=False):
            mem_size = (context_size // mem_freq) + (context_size % mem_freq > 0)
            mask = create_block_mask(
                create_mask_mod(context_size, window_size, mem_freq),
                None, None, context_size + mem_size, context_size + mem_size, device='cuda', BLOCK_SIZE=block_size,
                _compile=True
            )

            for res in benchmark(
                    flex_attention, context_size + mem_size,
                    {'impl': f'flex_window_mem_compr{mem_freq}', 'context_length': context_size},
                    enable_gqa=True, block_mask=mask
            ):
                data.append(res)
    except Exception as e:
        print('Exc!')
        print(str(e), e.__class__.__name__)


df = pd.DataFrame(data)
df.to_csv('perf_forward.csv')
