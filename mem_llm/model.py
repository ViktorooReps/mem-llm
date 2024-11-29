import json
from pathlib import Path
from typing import TypeVar, Type

import torch
from safetensors.torch import load_model, save_model
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _mask_mod_signature, \
    and_masks

from mem_llm.interface import Generator, Configurable, ModelOutput, Cache
from mem_llm.noop import Noop


torch._dynamo.config.cache_size_limit = 1000


flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)


_T = TypeVar('_T', bound=Configurable)


STR2DTYPE = {
    'float': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float64': torch.float64,
}

DTYPE2STR = {v: k for k, v in STR2DTYPE.items()}


def deserialize_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    return STR2DTYPE[dtype]


def serialize_dtype(dtype: str | torch.dtype) -> str:
    if isinstance(dtype, str):
        return dtype

    return DTYPE2STR[dtype]


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None, dtype: torch.dtype | None):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(torch.nn.Module):

    def __init__(self, dims: int, base: float, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.dims = dims
        self.base = base
        self.device = device
        self.dtype = dtype
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dims, 2, device=self.device).float() / self.dims))

    def forward(
            self,
            x: torch.Tensor,  # (B, L, H)
            x_pos: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        
        if x.shape[0] != 1:
            # FIXME: torch.outer will not work otherwise
            raise NotImplementedError() 

        t = x_pos.type_as(self.inv_freq).squeeze(0)
        freqs = torch.outer(t, self.inv_freq)

        cos_freq = freqs.cos().to(self.dtype)[None, :, None, :]
        sin_freq = freqs.sin().to(self.dtype)[None, :, None, :]

        # apply_rotary_emb(x, cos, sin)
        assert x.ndim == 4  # multihead attention

        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos_freq + x2 * sin_freq
        y2 = x1 * (-sin_freq) + x2 * cos_freq

        return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):

    def __init__(
            self,
            n_q_heads: int,
            n_kv_heads: int,
            head_dims: int,
            rotary_inv_freq_base: float,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dims = head_dims

        extras = {'device': device, 'dtype': dtype}

        self.hidden_dims = self.head_dims * self.n_q_heads
        self.kv_dims = self.head_dims * self.n_kv_heads
        self.gqa_enabled = (self.n_q_heads != self.n_kv_heads)

        self.q_proj = CastedLinear(self.hidden_dims, self.hidden_dims, **extras)
        self.k_proj = CastedLinear(self.hidden_dims, self.kv_dims, **extras)
        self.v_proj = CastedLinear(self.hidden_dims, self.kv_dims, **extras)

        # value residual lambda
        self.value_residual_coeff = nn.Parameter(torch.tensor(0.5))

        # rotary embeddings
        self.rotary = Rotary(self.head_dims, rotary_inv_freq_base, **extras)

        # output projection
        self.out_proj = CastedLinear(self.hidden_dims, self.hidden_dims, **extras)
        self.out_proj.weight.data.zero_()

    def forward(
            self,
            x: torch.Tensor,  # (B, L, H)
            x_pos: torch.Tensor,  # (B, L)
            block_mask: BlockMask | None
    ) -> torch.Tensor:

        # TODO: compute dense variant with cache

        batch_size, seq_length, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_length, self.n_q_heads, self.head_dims)
        k = self.k_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dims)
        v = self.v_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dims)

        q, k = norm(q), norm(k)
        q, k = self.rotary(q, x_pos), self.rotary(k, x_pos)

        # TODO: soft cap score mod
        y = flex_attention(
            q.transpose(1, 2).contiguous(), 
            k.transpose(1, 2).contiguous(), 
            v.transpose(1, 2).contiguous(),
            block_mask=block_mask,
            enable_gqa=self.gqa_enabled
        )
        y = y.transpose(1, 2).contiguous().view_as(x)  # re-assemble all head outputs side by side
        y = self.out_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, device: torch.device, dtype: torch.dtype):
        super().__init__()

        extras = {'device': device, 'dtype': dtype}

        self.hid_proj = CastedLinear(input_dims, hidden_dims, **extras)
        self.out_proj = CastedLinear(hidden_dims, input_dims, **extras)
        self.out_proj.weight.data.zero_()

    def forward(self, x):
        x = self.hid_proj(x)
        x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU;
        x = self.out_proj(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(
            self,
            n_q_heads: int,
            n_kv_heads: int,
            head_dims: int,
            mlp_hidden_dims_expansion: float,
            rotary_inv_freq_base: float,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()

        extras = {'device': device, 'dtype': dtype}

        self.hidden_dims = n_q_heads * head_dims
        self.mlp_hidden_dims = int(self.hidden_dims * mlp_hidden_dims_expansion)

        self.attn = CausalSelfAttention(n_q_heads, n_kv_heads, head_dims, rotary_inv_freq_base, **extras)
        self.mlp = MLP(self.hidden_dims, self.mlp_hidden_dims, **extras)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(
            self,
            x: torch.Tensor,  # (B, L, H)
            x_pos: torch.Tensor,  # (B, L)
            embed_residual: torch.Tensor | None,
            block_mask: BlockMask | None
    ) -> torch.Tensor:

        if embed_residual is not None:
            x = self.lambdas[0] * x + self.lambdas[1] * embed_residual
        
        x = x + self.attn(norm(x), x_pos, block_mask)
        x = x + self.mlp(norm(x))
        return x



def create_mem_window_mask_mod(
        pos: torch.Tensor,
        *,
        n_mem: int,
        mem_pad: int,
        global_window: int,
        local_window: int
) -> _mask_mod_signature:
    """
    Designed for the use on the input that consists of the concatenation of memory states and context states.
    Does not support padding! Pass in multiple sequences as concatenation of documents!

    Only works for square inputs! With asymmetric inputs use dense attention and evict non-relevant tokens from
    cache.

    pos: (L,)
    """
    main_start = n_mem + mem_pad


    def causal_window_mask_with_mem(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        q_pos = pos[q_idx]
        kv_pos = pos[kv_idx]

        # differentiator of main from mem parts
        is_mem_kv = (kv_idx < n_mem)
        is_mem_q = (q_idx < n_mem)

        # FIXME: remove padding once https://github.com/pytorch/pytorch/issues/139064 is resolved
        is_mem_pad_kv = ~is_mem_kv & (kv_idx < main_start)
        is_mem_pad_q = ~is_mem_q & (q_idx < main_start)

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


# FIXME: remove padding once https://github.com/pytorch/pytorch/issues/139064 is resolved
target_n_mem_padded = None


class MemLLM(Generator, Configurable):
    """
    Largely inspired by https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py

    Added mem tokens to sparsely scale to larger context windows. Does not support batching!!
    Process multiple sequences by concatenating them with <EOS>

    TODO: implement inference (caching)
    """
    def __init__(
            self,
            vocab_size: int = 128,
            n_q_heads: int = 8,
            n_kv_heads: int = 4,
            head_dims: int = 128,
            mlp_hidden_dims_expansion: float = 2.0,
            rotary_inv_freq_base: float = 10_000.0,
            n_layers: int = 24,
            local_window: int = 128,
            global_window: int = 128,
            mem_freq: int | None = 64,  # None for simple window attention
            logit_soft_cap: float | None = None,  # 30.0
            attention_score_soft_cap: float | None = None,
            dtype: str | torch.dtype | None = torch.bfloat16,
            device: str | torch.device | None = 'cuda' if torch.cuda.is_available() else 'cpu',
            unet_design: bool = False,
            embeds_residual: bool = False,
            eos_token: int = 127,
            do_compile: bool = True,
    ):
        super(MemLLM, self).__init__()

        self.vocab_size = vocab_size
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dims = head_dims
        self.mlp_hidden_dims_expansion = mlp_hidden_dims_expansion
        self.rotary_inv_freq_base = rotary_inv_freq_base
        self.n_layers = n_layers
        self.local_window = local_window
        self.global_window = global_window
        self.mem_freq = mem_freq
        self.logit_soft_cap = logit_soft_cap
        self.attention_score_soft_cap = attention_score_soft_cap
        self.dtype = deserialize_dtype(dtype)
        self.device = torch.device(device)
        self.unet_design = unet_design
        self.embeds_residual = embeds_residual
        self.eos_token = eos_token
        self.do_compile = do_compile

        self.hidden_dims = n_q_heads * head_dims

        extras = {'device': self.device, 'dtype': self.dtype}

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dims, **extras)
        self.mem_embedding = nn.Parameter(torch.zeros(self.hidden_dims, **extras))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                n_q_heads=self.n_q_heads,
                n_kv_heads=self.n_kv_heads,
                head_dims=self.head_dims,
                mlp_hidden_dims_expansion=self.mlp_hidden_dims_expansion,
                rotary_inv_freq_base=self.rotary_inv_freq_base,
                **extras
            )
            for _ in range(self.n_layers)
        ])

        # in case of U-Net design, add skip connections from encoder to decoder
        if self.unet_design:
            self.num_encoder_layers = self.n_layers // 2
            self.num_decoder_layers = self.n_layers - self.num_encoder_layers

            self.unet_skip_weights = nn.Parameter(torch.ones(self.num_encoder_layers, **extras))
        else:
            self.num_encoder_layers = self.n_layers
            self.num_decoder_layers = 0

            self.unet_skip_weights = None

        self.lm_head = CastedLinear(self.hidden_dims, self.vocab_size, **extras)
        self.lm_head.weight.data.zero_()

    # These setters are used for attention window warmup (during training, we increase the window gradually)
    # You can also use them to manage inference speed and cache memory consumption.

    def set_local_window(self, local_window: int):
        self.local_window = local_window

    def set_global_window(self, global_window: int):
        self.global_window = global_window

    def to_config(self):
        return {
            'vocab_size': self.vocab_size,
            'n_q_heads': self.n_q_heads,
            'n_kv_heads': self.n_kv_heads,
            'head_dims': self.head_dims,
            'mlp_hidden_dims_expansion': self.mlp_hidden_dims_expansion,
            'rotary_inv_freq_base': self.rotary_inv_freq_base,
            'n_layers': self.n_layers,
            'local_window': self.local_window,
            'global_window': self.global_window,
            'mem_freq': self.mem_freq,
            'dtype': serialize_dtype(self.dtype),
            'device': str(self.device),
            'unet_design': self.unet_design,
            'embeds_residual': self.embeds_residual,
            'eos_token': self.eos_token,
            'do_compile': self.do_compile,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        assert path.is_dir()

        config_path = path / 'config.json'
        weights_path = path / 'weights.safetensors'

        super().save(config_path)  # saves config
        save_model(self, str(weights_path.absolute()))

    @classmethod
    def load(cls: Type[_T], path: str | Path, *, device: int | str | torch.device = 'cpu') -> _T:
        path = Path(path)

        config_path = path / 'config.json'
        weights_path = path / 'weights.safetensors'

        with open(config_path, 'r') as f:
            config = json.load(f)

        model = cls.from_config(config, device=device)
        load_model(model, weights_path, strict=True, device=device)

        return model

    def forward(
            self,
            tokens: torch.Tensor,  # (L,)
            past_cache: Cache | None = None,
            *,
            return_cache: bool = False
    ) -> ModelOutput:

        assert len(tokens.shape) == 1

        seq_length = len(tokens)
        cache_size = 0 if past_cache is None else seq_length

        # FIXME: in-document masking will not work with cache!
        # FIXME: this method works with sparse attention on square inputs (#KV = #Q),
        #  which is not compatible for generation with cache
        if cache_size or return_cache:
            raise NotImplementedError

        eos_mask: torch.Tensor = (tokens == self.eos_token)  # noqa
        doc_ids_per_token = torch.cumsum(eos_mask, dim=0)

        eos_positions = torch.nonzero(eos_mask, as_tuple=True)[0]
        doc_end_indices = torch.concat([eos_positions, eos_positions.new_tensor([seq_length])], dim=0)
        doc_start_indices = torch.concat([eos_positions.new_tensor([0]), eos_positions], dim=0)

        doc_lengths = doc_end_indices - doc_start_indices
        doc_shifts_per_token = doc_start_indices.repeat_interleave(doc_lengths, output_size=seq_length)

        absolute_token_positions = torch.arange(seq_length, device=doc_end_indices.device, dtype=torch.int)
        document_positions = absolute_token_positions - doc_shifts_per_token

        if self.mem_freq is not None:
            is_mem = ((document_positions % self.mem_freq) == 0)
            mem_pos = document_positions[is_mem]
            mem_doc_ids = doc_ids_per_token[is_mem]
        else:
            mem_pos = document_positions.new_empty(0)
            mem_doc_ids = doc_ids_per_token.new_empty(0)

        n_mem = len(mem_pos)

        # FIXME: FlexAttention is horrible with dynamic shapes https://github.com/pytorch/pytorch/issues/139064,
        #  so we pad here up to block size to hopefully avoid different shapes during training. But this will not work
        #  on every input, I've only tested it on FineWeb-Edu.
        global target_n_mem_padded
        if self.mem_freq is None:
            target_n_mem_padded = 0

        if target_n_mem_padded is None:
            mem_pad = 128 - n_mem % 128
            mem_pad += 256
            target_n_mem_padded = mem_pad + n_mem
        else:
            mem_pad = target_n_mem_padded - n_mem

        full_positions = torch.concat([
            mem_pos, mem_pos.new_zeros(mem_pad), document_positions
        ], dim=0)
        full_doc_ids = torch.concat([
            mem_doc_ids, mem_doc_ids.new_zeros(mem_pad), doc_ids_per_token
        ], dim=0).contiguous()
        full_eos_mask = torch.concat([
            eos_mask.new_ones(n_mem), eos_mask.new_zeros(mem_pad), eos_mask
        ], dim=0).contiguous()

        def document_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
            # attend only within the same document and do not attend to and from <eos> tokens
            return ((full_doc_ids[q_idx] == full_doc_ids[kv_idx])
                    & ~full_eos_mask[q_idx]
                    & ~full_eos_mask[kv_idx])

        block_mask = create_block_mask(
            and_masks(document_mask, create_mem_window_mask_mod(
                pos=full_positions,
                n_mem=n_mem,
                mem_pad=mem_pad,
                local_window=self.local_window,
                global_window=self.global_window,
            )),
            None,  None, seq_length + n_mem + mem_pad, seq_length + n_mem + mem_pad,
            device=str(tokens.device), _compile=self.do_compile
        )

        x = self.token_embedding(tokens)
        mem = self.mem_embedding.unsqueeze(0).repeat(n_mem + mem_pad, 1)  # FIXME: remove padding once https://github.com/pytorch/pytorch/issues/139064 is resolved

        x = torch.concat([mem, x], dim=0)  # the first n_mem tokens are memory
        x = norm(x)

        # inner modules work with batched data
        x = x.unsqueeze(0)
        x_pos = full_positions.unsqueeze(0)

        embeds = None
        if self.embeds_residual:
            embeds = x

        # Store outputs for U-Net-style skip connections
        # If there is no need to store them, just ignore any actions with the list with Noop
        skip_connections = [] if self.unet_design else Noop()

        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.transformer_blocks[i](x, x_pos, embeds, block_mask)
            skip_connections.append(x)

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.unet_skip_weights[i] * skip_connections.pop()
            x = self.transformer_blocks[self.num_encoder_layers + i](x, x_pos, embeds, block_mask)

        # drop memory for logits computation
        x = x[:, n_mem + mem_pad:]

        x = norm(x)
        logits = self.lm_head(x)

        if self.logit_soft_cap is not None:
            logits = self.logit_soft_cap * torch.tanh(logits / self.logit_soft_cap)

        return ModelOutput(logits.squeeze(0), None)
