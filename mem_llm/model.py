import json
from pathlib import Path
from typing import TypeVar, Type

import torch
from safetensors.torch import load_model, save_model
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from torch.nn.functional import scaled_dot_product_attention

from mem_llm.custom_logging import logger
from mem_llm.interface import Generator, ConfigurableMixin, ModelOutput, Cache, WindowedMixin
from mem_llm.masking import create_mem_block_masks
from mem_llm.noop import Noop


# for some reason FlexAttention does not work without this
torch._dynamo.config.cache_size_limit = 1000


DO_COMPILE = False

if DO_COMPILE:
    flex_attention = torch.compile(flex_attention, dynamic=False)

_T = TypeVar('_T', bound=ConfigurableMixin)


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
    return F.rms_norm(x, (x.size(-1),), eps=1e-5)


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
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dims, 2, device=self.device).double() / self.dims))

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


class MLP(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, device: torch.device, dtype: torch.dtype):
        super().__init__()

        extras = {'device': device, 'dtype': dtype}

        self.hid_proj = CastedLinear(input_dims, hidden_dims, **extras)
        self.out_proj = CastedLinear(hidden_dims, input_dims, **extras)
        self.out_proj.weight.data.zero_()

    def forward(self, x):
        x = self.hid_proj(x)
        x = F.silu(x)
        x = self.out_proj(x)
        return x


class MemTransformerBlock(nn.Module):
    """
    Transformer block with support for memory tokens. Memory tokens are always at the start of the sequence. You can
    set the actual position of memory tokens via x_pos argument.

    The memory attention logic should be implemented either in BlockMasks passed to the forward method,
    or via "state eviction" and applying dense attention one token at a time (or 2 if a memory token is also included).
    """

    def __init__(
            self,
            n_q_heads: int,
            n_kv_heads: int,
            head_dims: int,
            mlp_hidden_dims_expansion: float,
            rotary_inv_freq_base: float,
            precompute_mem: bool,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()

        extras = {'device': device, 'dtype': dtype}

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dims = head_dims

        self.hidden_dims = self.head_dims * self.n_q_heads
        self.kv_dims = self.head_dims * self.n_kv_heads
        self.gqa_enabled = (self.n_q_heads != self.n_kv_heads)
        self.mlp_hidden_dims = int(self.hidden_dims * mlp_hidden_dims_expansion)
        self.precompute_mem = precompute_mem

        self.q_proj = CastedLinear(self.hidden_dims, self.hidden_dims, **extras)
        self.k_proj = CastedLinear(self.hidden_dims, self.kv_dims, **extras)
        self.v_proj = CastedLinear(self.hidden_dims, self.kv_dims, **extras)

        if self.precompute_mem:
            self.k_mem_proj = CastedLinear(self.hidden_dims, self.kv_dims, **extras)
            self.v_mem_proj = CastedLinear(self.hidden_dims, self.kv_dims, **extras)
        else:
            self.k_mem_proj = None
            self.v_mem_proj = None

        # rotary embeddings
        self.rotary = Rotary(self.head_dims, rotary_inv_freq_base, **extras)

        # output projection
        self.out_proj = CastedLinear(self.hidden_dims, self.hidden_dims, **extras)
        self.out_proj.weight.data.zero_()

        self.mlp = MLP(self.hidden_dims, self.mlp_hidden_dims, **extras)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(
            self,
            x: torch.Tensor,  # (B, n_mem + L, H)
            x_pos: torch.Tensor,  # (B, n_mem + L)
            n_mem: int,
            embed_residual: torch.Tensor | None,  # (B, n_mem + L, H)
            block_mask: BlockMask | None,
            mem_block_mask: BlockMask | None,
    ) -> torch.Tensor:

        batch_size, seq_length, _ = x.shape

        if embed_residual is not None:
            x = self.lambdas[0] * x + self.lambdas[1] * embed_residual

        k = self.k_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dims)
        v = self.v_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dims)

        k = self.rotary(norm(k), x_pos)

        # TODO: add caching here

        if self.precompute_mem:
            # prerun the block on the memory
            x_mem = x[:, :n_mem]
            x_mem_pos = x_pos[:, :n_mem]

            x_mem = self._run_block(x_mem, x_mem_pos, k, v, block_mask=mem_block_mask)

            # update KV with precomputed mem
            k_mem = self.k_mem_proj(x_mem).view(batch_size, n_mem, self.n_kv_heads, self.head_dims)
            v_mem = self.v_mem_proj(x_mem).view(batch_size, n_mem, self.n_kv_heads, self.head_dims)

            k_mem = self.rotary(norm(k_mem), x_mem_pos)

            k_main = k[:, n_mem:]
            v_main = v[:, n_mem:]

            k = torch.concat([k_mem, k_main], dim=1)
            v = torch.concat([v_mem, v_main], dim=1)

            # process the rest of the inputs normally
            x = x[:, n_mem:]
            x_pos = x_pos[:, n_mem:]

        x = self._run_block(x, x_pos, k, v, block_mask=block_mask)

        if self.precompute_mem:
            x = torch.concat([x_mem, x], dim=1)

        return x

    def _run_block(
            self,
            x: torch.Tensor,  # (B, Q, H)
            x_pos: torch.Tensor,  # (B, Q)
            k: torch.Tensor,  # (B, KV, kvh, h) - precomputed k (possibly cached)
            v: torch.Tensor,  # (B, KV, kvh, h) - precomputed v (possibly cached)
            block_mask: BlockMask | None
    ) -> torch.Tensor:  # (B, Q, H)
        """
        Runs a transformer block on x wrt to precomputed KV. If block_mask is not None, runs a
        sparse attention implementation (FlexAttention), otherwise SDPA will be used.

        NOTE: SDPA can only be used when Q=1
        """
        batch_size, n, _ = x.shape

        q = self.q_proj(x).view(batch_size, n, self.n_q_heads, self.head_dims)
        q = self.rotary(norm(q), x_pos)

        if block_mask is None:
            assert n == 1

            # run dense attention
            attn = scaled_dot_product_attention(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                is_causal=False,  # we force n == 1, and assume that causality is preserved by the user
                enable_gqa=self.gqa_enabled
            ).transpose(1, 2).contiguous().view(batch_size, n, self.hidden_dims)
        else:
            # run sparse attention
            attn = flex_attention(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                block_mask=block_mask,
                enable_gqa=self.gqa_enabled
            ).transpose(1, 2).contiguous().view(batch_size, n, self.hidden_dims)

        x = x + self.out_proj(attn)
        x = x + self.mlp(norm(x))

        return x


class MemLLM(Generator, ConfigurableMixin, WindowedMixin):
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
            rotary_inv_freq_base: float = 500_000.0,
            n_layers: int = 24,
            local_window: int = 128,
            global_window: int = 128,
            mem_freq: int | None = 64,  # None for simple window attention
            precompute_mem: bool = False,
            logit_soft_cap: float | None = None,  # 30.0
            attention_score_soft_cap: float | None = None,
            torch_dtype: str | torch.dtype | None = torch.bfloat16,
            device: str | torch.device | None = 'cuda' if torch.cuda.is_available() else 'cpu',
            unet_design: bool = False,
            embeds_residual: bool = False,
            eos_token_id: int = 127,
            do_compile: bool = True,
    ):
        super(MemLLM, self).__init__()

        if mem_freq is None and precompute_mem:
            logger.warning(
                f'Passed mem_freq={mem_freq} and precompute_mem={precompute_mem}. '
                f'Using simple window attention, so precompute_mem argument is ignored.'
            )
            precompute_mem = False

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
        self.precompute_mem = precompute_mem
        self.logit_soft_cap = logit_soft_cap
        self.attention_score_soft_cap = attention_score_soft_cap
        self.dtype = deserialize_dtype(torch_dtype)
        self.device = torch.device(device)
        self.unet_design = unet_design
        self.embeds_residual = embeds_residual
        self.eos_token = eos_token_id
        self.do_compile = do_compile

        self.hidden_dims = n_q_heads * head_dims

        extras = {'device': self.device, 'dtype': self.dtype}

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dims, **extras)
        self.mem_embedding = nn.Parameter(torch.zeros(self.hidden_dims, **extras))

        self.transformer_blocks = nn.ModuleList([
            MemTransformerBlock(
                n_q_heads=self.n_q_heads,
                n_kv_heads=self.n_kv_heads,
                head_dims=self.head_dims,
                mlp_hidden_dims_expansion=self.mlp_hidden_dims_expansion,
                rotary_inv_freq_base=self.rotary_inv_freq_base,
                precompute_mem=self.precompute_mem,
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

    def set_mem_freq(self, mem_freq: int):
        self.mem_freq = mem_freq

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
            'precompute_mem': self.precompute_mem,
            'dtype': serialize_dtype(self.dtype),
            'device': str(self.device),
            'unet_design': self.unet_design,
            'embeds_residual': self.embeds_residual,
            'eos_token_id': self.eos_token,
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
    def load(cls: Type[_T], path: str | Path, *, device: int | str | torch.device = 'cpu', **config_changes) -> _T:
        path = Path(path)

        config_path = path / 'config.json'
        weights_path = path / 'weights.safetensors'

        with open(config_path, 'r') as f:
            config = json.load(f)

        model = cls.from_config(config, device=device, **config_changes)
        load_model(model, weights_path, strict=True, device=device)

        return model

    def forward(
            self,
            tokens: torch.Tensor,  # (L,) or (B, L)
            past_cache: Cache | None = None,
            *,
            use_cache: bool = False,
            num_logits_to_keep: int | None = None
    ) -> ModelOutput:

        # INPUTS PREP ==========================================================================

        # TODO: for compatibility with HF this should be L - cache_size = 1
        # we use dense attention when the inputs are of length 1
        # this happens (for example) during generation where we process inputs one token at a time
        use_dense_attention = (tokens.shape[-1] == 1)

        batch_size = None
        if len(tokens.shape) == 2 and not use_dense_attention:
            # flatten the batch by treating each example as a separate document
            batch_size, _ = tokens.shape
            tokens = torch.concat([tokens, tokens.new_full((batch_size, 1), fill_value=self.eos_token)], dim=-1)
            tokens = tokens.view(-1)

        seq_length = tokens.shape[-1]
        cache_size = 0 if past_cache is None else seq_length

        # FIXME: in-document masking will not work with cache!
        # FIXME: this method works with sparse attention on square inputs (#KV = #Q),
        #  which is not compatible for generation with cache: use dense implementation in this case
        if cache_size or use_cache:
            raise NotImplementedError

        if not use_dense_attention:
            block_mask, mem_block_mask, n_mem, full_positions = create_mem_block_masks(
                tokens=tokens,
                eos_token=self.eos_token,
                mem_freq=self.mem_freq,
                local_window=self.local_window,
                global_window=self.global_window,
                separate_mem_and_main_update=self.precompute_mem,
                do_compile=self.do_compile
            )
            x_pos = full_positions.unsqueeze(0)
        else:
            block_mask = None
            mem_block_mask = None
            new_position = 0  # TODO: cache_end_position + 1
            n_mem = int((new_position % self.mem_freq) == 0)
            x_pos = tokens.new_tensor([[new_position] * (n_mem + 1)])  # TODO: determine from cache

        if batch_size is None:
            # inner modules work with batched data
            tokens = tokens.unsqueeze(0)

        # TRANSFORMER PASS =====================================================================

        x = self.token_embedding(tokens)
        mem = self.mem_embedding.view(1, 1, -1).repeat(1, n_mem, 1)

        x = torch.concat([mem, x], dim=1)  # the first n_mem tokens are memory
        x = norm(x)

        embeds = None
        if self.embeds_residual:
            embeds = torch.clone(x)

        # Store outputs for U-Net-style skip connections
        # If there is no need to store them, just ignore any actions with the list with Noop
        skip_connections = [] if self.unet_design else Noop()

        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.transformer_blocks[i](
                x, x_pos,
                embed_residual=embeds,
                n_mem=n_mem,
                block_mask=block_mask,
                mem_block_mask=mem_block_mask
            )
            skip_connections.append(x)

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.unet_skip_weights[i] * skip_connections.pop()
            x = self.transformer_blocks[self.num_encoder_layers + i](
                x, x_pos,
                embed_residual=embeds,
                n_mem=n_mem,
                block_mask=block_mask,
                mem_block_mask=mem_block_mask
            )

        # drop memory for logits computation
        x = x[:, n_mem:]

        if batch_size is not None and not use_dense_attention:
            # return to original shape
            x = x.view(batch_size, -1, self.hidden_dims)
            # drop added eos token
            x = x[:, :-1, :]

        x = norm(x)
        logits = self.lm_head(x[: -num_logits_to_keep:] if num_logits_to_keep is not None else x)

        if self.logit_soft_cap is not None:
            logits = self.logit_soft_cap * torch.tanh(logits / self.logit_soft_cap)

        return ModelOutput(
            logits=logits.squeeze(0) if batch_size is None else logits,
            cache=None  # TODO
        )
