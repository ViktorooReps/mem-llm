from typing import Literal

import torch
from torch.nn.attention.flex_attention import create_block_mask, _mask_mod_signature, BlockMask, and_masks

from mem_llm.custom_logging import logger

DO_COMPILE = False

if DO_COMPILE:
    create_block_mask = torch.compile(create_block_mask, dynamic=False)


def create_mem_window_mask_mod(
        pos: torch.Tensor,
        *,
        n_mem: int,
        mem_pad: int,
        global_window: int,
        local_window: int,
        kind: Literal['all', 'mem_update', 'main_update']
) -> _mask_mod_signature:
    """
    Designed for the use on the input that consists of the concatenation of memory states and context states.
    Does not support padding! Pass in multiple sequences as concatenation of documents!

    Only works for square inputs! With asymmetric inputs use dense attention and evict non-relevant tokens from
    cache.

    pos: (L,)
    """
    usual_main_start = n_mem + mem_pad
    main_start_kv = usual_main_start  # KV always has both mem and main tokens

    # depending on the kind of the update, we have different inputs
    # 'all': update both mem and main tokens, so the input is square
    #   Q_LEN = KV_LEN = seq_len + n_mem + mem_pad
    if kind == 'all':
        main_start_q = usual_main_start

    # 'mem_update': update only mem tokens, so the queries are memory, key-values are memory and main tokens:
    #   Q_LEN = n_mem + mem_pad
    #   KV_LEN = seq_len + n_mem + mem_pad
    if kind == 'mem_update':
        main_start_q = usual_main_start  # since Q_LEN = usual_main_start, #main_q = 0

    # 'mem_update': update only main tokens
    #   Q_LEN = seq_len
    #   KV_LEN = seq_len + n_mem + mem_pad
    if kind == 'main_update':
        main_start_q = 0

    def causal_window_mask_with_mem(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        # The start is shifted in cases where main_start_q = 0.
        # This causes later is_mem_q to always be False,
        #    as well as is_mem_pad_q to be always False.
        # So practically, we just restrict queries to main tokens
        corrected_q_idx = q_idx + usual_main_start - main_start_q

        q_pos = pos[corrected_q_idx]
        kv_pos = pos[kv_idx]

        # differentiator of main from mem parts
        is_mem_kv = (kv_idx < n_mem)
        is_mem_q = (corrected_q_idx < n_mem)

        # FIXME: remove padding once https://github.com/pytorch/pytorch/issues/139064 is resolved
        is_mem_pad_kv = ~is_mem_kv & (kv_idx < main_start_kv)
        is_mem_pad_q = ~is_mem_q & (corrected_q_idx < main_start_q)

        causal = (kv_pos <= q_pos)

        window_local = (q_pos - kv_pos < local_window)
        window_global = (q_pos - kv_pos < global_window)  # for attending to mem

        # special case for mem2main:
        # 1. We don't want mem at position 0 to have any information
        # 2. If we consider mem as "relay", there is no need to relay info from token x to itself

        # do not include the diagonal!
        causal_mem2main = (kv_pos < q_pos)
        # we did not include the diagonal, so add 1 token more here
        window_mem2main = (q_pos - kv_pos <= local_window)

        case_main2main = (~is_mem_kv & ~is_mem_q & causal & window_local)
        case_main2mem = (is_mem_kv & ~is_mem_q & causal & window_global)
        case_mem2mem = (is_mem_kv & is_mem_q & causal & window_global)
        case_mem2main = (~is_mem_kv & is_mem_q & causal_mem2main & window_mem2main)
        return (case_main2main | case_main2mem | case_mem2mem | case_mem2main) & ~is_mem_pad_kv & ~is_mem_pad_q

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

    return doc_ids_per_token, document_positions


target_n_mem_padded = None


def create_mem_block_masks(
        tokens: torch.Tensor,
        *,
        eos_token: int,
        mem_freq: int | None = None,
        local_window: int | None = None,
        global_window: int | None = None,
        separate_mem_and_main_update: bool = False,
        do_compile: bool = True,
        pad_memory: bool = False,
) -> (BlockMask, BlockMask | None, int, torch.Tensor):

    with_batch_dim = False
    if len(tokens.shape) > 1:
        with_batch_dim = True
        assert tokens.shape[0] == 1
        tokens = tokens.squeeze(0)

    seq_length = len(tokens)

    if mem_freq is None and separate_mem_and_main_update:
        logger.warning('Possible configuration error: separate_mem_and_main_update=True and mem_freq is not set')
        separate_mem_and_main_update = False

    if local_window is None:
        local_window = seq_length
    if global_window is None:
        global_window = seq_length

    eos_mask: torch.Tensor = (tokens == eos_token)  # noqa
    doc_ids_per_token, document_positions = get_document_info(eos_mask)

    if mem_freq is not None:
        is_mem = ((document_positions % mem_freq) == 0)
    else:
        is_mem = torch.zeros_like(eos_mask)

    mem_pos = document_positions[is_mem]
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
        return (full_doc_ids[q_idx] == full_doc_ids[kv_idx]) & ~full_eos_mask[q_idx] & ~full_eos_mask[kv_idx]

    def document_mask_main_update(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        # queries are only main, so we need to shift q_idx accordingly to skip memory
        return base_document_mask(b, h, q_idx + n_mem + mem_pad, kv_idx)

    # mem is at the start, so we don't need to shift anything
    document_mask_mem_update = base_document_mask

    # default state
    # square block matrix, both updates - main and mem - at once
    main_block_mask_kind: Literal['all', 'main_update'] = 'all'
    n_updated = seq_length + n_mem + mem_pad
    document_mask = base_document_mask
    mem_block_mask = None

    if separate_mem_and_main_update:
        mem_block_mask = create_block_mask(
            and_masks(document_mask_mem_update, create_mem_window_mask_mod(
                pos=full_positions,
                n_mem=n_mem,
                mem_pad=mem_pad,
                local_window=local_window,
                global_window=global_window,
                kind='mem_update',
            )),
            B=None,
            H=None,
            Q_LEN=n_mem + mem_pad,  # update for mem only, so queries are all memory
            KV_LEN=seq_length + n_mem + mem_pad,
            device=str(tokens.device),
            _compile=do_compile
        )

        # switch the main mask to main-only update
        main_block_mask_kind = 'main_update'
        document_mask = document_mask_main_update
        n_updated = seq_length

    block_mask = create_block_mask(
        and_masks(document_mask, create_mem_window_mask_mod(
            pos=full_positions,
            n_mem=n_mem,
            mem_pad=mem_pad,
            local_window=local_window,
            global_window=global_window,
            kind=main_block_mask_kind
        )),
        B=None,
        H=None,
        Q_LEN=n_updated,
        KV_LEN=seq_length + n_mem + mem_pad,
        device=str(tokens.device),
        _compile=do_compile
    )

    full_positions = full_positions.unsqueeze(0) if with_batch_dim else full_positions
    return block_mask, mem_block_mask, n_mem + mem_pad, full_positions
