from typing import Literal

import torch
from torch.nn.attention.flex_attention import create_block_mask, _mask_mod_signature, BlockMask, and_masks

from mem_llm.custom_logging import logger

DO_COMPILE = False

if DO_COMPILE:
    create_block_mask = torch.compile(create_block_mask, dynamic=False, mode="max-autotune-no-cudagraphs")


def create_mem_window_mask_mod(
        pos: torch.Tensor,
        *,
        n_mem: int,
        mem_pad: int,
        global_window: int,
        local_window: int,
) -> _mask_mod_signature:
    """
    Designed for the use on the input that consists of the concatenation of memory states and context states.
    Does not support padding! Pass in multiple sequences as concatenation of documents!

    Only works for square inputs! With asymmetric inputs use dense attention and evict non-relevant tokens from
    cache.

    pos: (L,)
    """
    total_mem = n_mem + mem_pad

    def causal_window_mask_with_mem(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        q_pos = pos[q_idx]
        kv_pos = pos[kv_idx]

        # differentiator of main from mem parts
        is_mem_kv = (kv_idx < n_mem)
        is_mem_q = (q_idx < n_mem)

        # FIXME: remove padding once https://github.com/pytorch/pytorch/issues/139064 is resolved
        is_mem_pad_kv = ~is_mem_kv & (kv_idx < total_mem)
        is_mem_pad_q = ~is_mem_q & (q_idx < total_mem)

        causal = (kv_pos <= q_pos)

        window_local = (q_pos - kv_pos < local_window)
        window_global = (q_pos - kv_pos < global_window)  # for attending to mem

        case_kv_main = (~is_mem_kv & causal & window_local)
        case_kv_mem = (is_mem_kv & causal & window_global)
        return (case_kv_main | case_kv_mem) & ~is_mem_pad_kv & ~is_mem_pad_q

    return causal_window_mask_with_mem


def get_document_info(eos_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Returns document ids and relative position of tokens in the document"""
    seq_length = len(eos_mask)

    # we include <eos> token into the ending document
    # tokens: <t_0> <t_1> <eos> <t_0> <eos>
    # doc id:  0     0     0     1     1
    doc_ids_per_token = torch.cumsum(eos_mask, dim=0).roll(1)
    doc_ids_per_token[0] = 0

    # here we add +1 to eos_positions to translate to doc_end_indices to reflect the fact above
    eos_positions = torch.nonzero(eos_mask, as_tuple=True)[0]
    if not eos_mask[-1]:
        doc_end_indices = torch.concat([eos_positions + 1, eos_positions.new_tensor([seq_length])], dim=0)
    else:
        doc_end_indices = eos_positions + 1

    doc_start_indices = doc_end_indices.roll(1)
    doc_start_indices[0] = 0

    doc_lengths = doc_end_indices - doc_start_indices
    doc_shifts_per_token = doc_start_indices.repeat_interleave(doc_lengths, output_size=seq_length)

    absolute_token_positions = torch.arange(seq_length, device=doc_end_indices.device, dtype=torch.int)
    document_positions = absolute_token_positions - doc_shifts_per_token

    # reserve 0 position for bos token
    # this way the mem token at the start of the document (bos)
    # can only attend to itself
    return doc_ids_per_token, document_positions + 1


target_n_mem_padded = None


def create_mem_block_masks(
        tokens: torch.Tensor,
        *,
        eos_token: int,
        mem_freq: int | None = None,
        local_window: int | None = None,
        global_window: int | None = None,
        do_compile: bool = True,
        pad_memory: bool = False,
) -> (BlockMask, int, torch.Tensor):

    with_batch_dim = False
    if len(tokens.shape) > 1:
        with_batch_dim = True
        assert tokens.shape[0] == 1
        tokens = tokens.squeeze(0)

    seq_length = len(tokens)

    if local_window is None:
        local_window = seq_length
    if global_window is None:
        global_window = seq_length

    eos_mask: torch.Tensor = (tokens == eos_token)  # noqa
    doc_ids_per_token, document_positions = get_document_info(eos_mask)

    if mem_freq is not None:
        # document positions start with 1
        is_mem = ((document_positions % mem_freq) == 1)
    else:
        is_mem = torch.zeros_like(eos_mask)

    mem_pos = document_positions[is_mem] - 1
    mem_doc_ids = doc_ids_per_token[is_mem]

    n_mem = len(mem_pos)

    # FIXME: FlexAttention is horrible with dynamic shapes https://github.com/pytorch/pytorch/issues/139064,
    #  so we pad here up to block size to hopefully avoid different shapes during training. But this will not work
    #  on every input, I've only tested it on FineWeb-Edu, where examples are fairly long.
    global target_n_mem_padded
    if mem_freq is None or not pad_memory:
        target_n_mem_padded = n_mem

    if target_n_mem_padded is None:
        mem_pad = 128 - n_mem % 128
        mem_pad += 256
        target_n_mem_padded = mem_pad + n_mem
    else:
        mem_pad = target_n_mem_padded - n_mem

    if mem_pad:
        mem_pos = torch.concat([mem_pos, mem_pos.new_zeros(mem_pad)])
        mem_doc_ids = torch.concat([mem_doc_ids, mem_doc_ids.new_zeros(mem_pad)])

    full_positions = torch.concat([mem_pos, document_positions], dim=0).contiguous()
    full_doc_ids = torch.concat([mem_doc_ids, doc_ids_per_token], dim=0).contiguous()
    full_eos_mask = torch.concat([eos_mask.new_zeros(n_mem + mem_pad), eos_mask], dim=0).contiguous()

    def base_document_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        # attend only within the same document and do not attend to and from <eos> tokens
        # allow <eos> only attend to itself
        return ((full_doc_ids[q_idx] == full_doc_ids[kv_idx])
                & ((full_eos_mask[q_idx] & full_eos_mask[kv_idx]) | (~full_eos_mask[q_idx] & ~full_eos_mask[kv_idx])))

    block_mask = create_block_mask(
        and_masks(base_document_mask, create_mem_window_mask_mod(
            pos=full_positions,
            n_mem=n_mem,
            mem_pad=mem_pad,
            local_window=local_window,
            global_window=global_window,
        )),
        B=None,
        H=None,
        Q_LEN=seq_length + n_mem + mem_pad,
        KV_LEN=seq_length + n_mem + mem_pad,
        device=str(tokens.device),
        _compile=do_compile
    )

    full_positions = full_positions.unsqueeze(0) if with_batch_dim else full_positions
    return block_mask, n_mem + mem_pad, full_positions
