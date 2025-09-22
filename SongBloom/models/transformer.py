from functools import reduce, partial
from packaging import version

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import autocast
from typing import Callable, Literal
import os, sys
import warnings
from torch.utils import checkpoint
from transformers.utils import is_flash_attn_2_available

try:
    assert is_flash_attn_2_available()
    assert torch.cuda.get_device_capability(torch.device("cuda")) >= (8, 0)
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, unpad_input, pad_input
    assert os.environ.get("DISABLE_FLASH_ATTN",'0') != "1"
except Exception as e:
    flash_attn_kvpacked_func = None
    flash_attn_func = None
    warnings.warn("Not support flash-attn!")

try:
    import natten
except ImportError:
    natten = None

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
        else:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

            self.scale_base = scale_base
            self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)
        seq_len = t.shape[0]

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

class RotaryEmbedding2D(RotaryEmbedding):
    def __init__(self, dim, w, **kwargs):
        super().__init__(dim // 2, **kwargs)
        self.w = w


    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device
        assert seq_len % self.w == 0 , f"{seq_len} % {self.w} != 0"
        h_len = seq_len // self.w

        t_h = torch.arange(h_len, device = device)
        t_w = torch.arange(self.w, device = device)

        return self.forward(t_h, t_w)

    @autocast('cuda', enabled = False)
    def forward(self, t_h: torch.Tensor, t_w: torch.Tensor):
        repeat_t_h = t_h.repeat_interleave(t_w.shape[0], dim=0)
        repeat_t_w = t_w.repeat(t_h.shape[0])
        freq_h, scale_h = super().forward(repeat_t_h)
        freq_w, scale_w = super().forward(repeat_t_w)
        freq = torch.stack([freq_h, freq_w], dim=-1) #h*w, D//2, 2
        freq = torch.cat(torch.unbind(freq, dim=-2), dim=-1)

        if self.scale is None:
            scale = 1.
        else:
            scale = torch.stack([scale_h, scale_w], dim=-1)
            scale = torch.cat(torch.unbind(scale, dim=-2), dim=-1)

        return freq, scale




def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)
    return torch.cat((t, t_unrotated), dim = -1)

# norms
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))


    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
        bias = False,
    ):
        super().__init__()
        self.act = activation
        self.up_proj = nn.Linear(dim_in, dim_out, bias=bias) if not use_conv else nn.Conv1d(dim_in, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.gate_proj = nn.Linear(dim_in, dim_out, bias=bias) if not use_conv else nn.Conv1d(dim_in, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            gate = self.gate_proj(x)
            x = self.up_proj(x)
            x = rearrange(x, 'b d n -> b n d')
            gate = rearrange(gate, 'b d n -> b n d')
        else:
            gate = self.gate_proj(x)
            x = self.up_proj(x)

        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        dim_ff = None,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
    ):
        super().__init__()
        inner_dim = dim_ff if dim_ff is not None else 4 * dim

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation, bias=not no_bias)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm: Literal['l2', 'ln', 'none'] = 'none',
        natten_kernel_size = None
    ):
        super().__init__() # <-- BUG FIX: ADDED THIS LINE
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim

        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is not None:
            return

        self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None

        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None
    ):
        # =====================================================================
        # ========= Start: This part is the same as the original ==============
        # =====================================================================
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln":
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        input_mask = context_mask # cross-attn
        if input_mask is None and not has_context: # self-attn
            input_mask = mask

        # determine masking
        masks = []
        final_attn_mask = None # The mask that will be applied to the attention matrix, taking all masks into account

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal
        if n == 1 and causal:
            causal = False

        # =====================================================================
        # === MODIFIED PART: Forcing manual attention for AMD compatibility ===
        # This section replaces the optimized FlashAttention and PyTorch SDP
        # with a manual implementation that is more stable on ROCm.
        # =====================================================================

        if h != kv_h:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = h // kv_h
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        scale = 1. / (q.shape[-1] ** 0.5)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if final_attn_mask is not None:
            dots = dots.masked_fill(~final_attn_mask, mask_value)

        if causal:
            # Using the globally defined `create_causal_mask` function
            causal_mask = create_causal_mask(i, j, device = device)
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = F.softmax(dots, dim=-1, dtype=torch.float32)
        attn = attn.type(dtype)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # =====================================================================
        # ========= Start: This part is the same as the original ==============
        # =====================================================================

        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        out = self.to_out(out)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):

        super().__init__()

        self.dim = dim

        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b n d')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            remove_norms = False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):

        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal

        self.pre_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()

        self.self_attn = Attention(
            dim,
            dim_heads = dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(
                dim,
                dim_heads = dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )

        self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)

        self.layer_ix = layer_ix

        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_cond_dim, dim * 6, bias=False)
            )

            nn.init.zeros_(self.to_scale_shift_gate[1].weight)
            #nn.init.zeros_(self.to_scale_shift_gate_self[1].bias)

    def forward(
        self,
        x,
        mask = None,
        global_cond=None,
        context = None,
        context_mask = None,
        rotary_pos_emb = None
    ):
        if self.global_cond_dim is not None:
            assert global_cond is not None
            # scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = checkpoint(self.to_scale_shift_gate, global_cond).unsqueeze(1).chunk(6, dim = -1)
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim = -1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
            x = x * torch.sigmoid(1 - gate_self)
            x = x + residual

            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            if self.conformer is not None:
                x = x + self.conformer(x)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual

        else:
            assert global_cond is None
            x = x + self.self_attn(self.pre_norm(x), mask = mask, rotary_pos_emb = rotary_pos_emb)

            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            if self.conformer is not None:
                x = x + self.conformer(x)

            x = x + self.ff(self.ff_norm(x))

        return x

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cross_atten_layer_idx=None,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        pos_emb_2d_size=1,
        rotary_base_val=10000,
        init_std=0.02,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            if pos_emb_2d_size == 1:
                self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32), base=rotary_base_val)
            else:
                self.rotary_pos_emb = RotaryEmbedding2D(max(dim_heads // 2, 32), pos_emb_2d_size, base=rotary_base_val)
        else:
            self.rotary_pos_emb = None

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            if pos_emb_2d_size != 1:
                raise NotImplementedError
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            if pos_emb_2d_size != 1:
                raise NotImplementedError
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)


        if cross_atten_layer_idx is None:
            cross_atten_layer_idx = list(range(depth))
        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend and (i in cross_atten_layer_idx),
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )

        self.apply(partial(self._init_weights,init_std=init_std))

    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        # Iterate over the transformer layers
        context, context_mask = kwargs.pop('context', None), kwargs.pop("context_mask", None)

        for layer_idx, layer in enumerate(self.layers):
            if layer.cross_attend:
                x = layer(x, mask, global_cond, context, context_mask, rotary_pos_emb=rotary_pos_emb, **kwargs)
            else:
                x = layer(x, mask, global_cond, rotary_pos_emb=rotary_pos_emb, **kwargs)
            if return_info:
                info["hidden_states"].append(x)

        x = self.project_out(x)

        if return_info:
            return x, info

        return x

    def _init_weights(self, module, init_std=0.02):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
