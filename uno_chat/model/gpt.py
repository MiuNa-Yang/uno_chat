import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    # base
    sequence_len: int = 1024
    n_embd: int = 768
    vocab_size: int = 50304
    n_layer: int = 12

    # attention
    n_q_head: int = 6
    n_kv_head: int = 6


def norm(x):
    """
    nn.RMSNorm
    :param x:
        - B, T, Embd_dim
    :return:
    """
    return F.rms_norm(
        input=x,
        normalized_shape=(x.size(-1),),  # 对最后一个纬度做归一化
    )


def apply_rotary_emb(x, cos, sin):
    """
    ROPE 旋转位置编码: 数学公式的等价实现方法
    """

    # x: [B, T, num_heads, head_dim]
    # cos, sin: [T, head_dim]

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]  # 将最后一个维度一分为二 [B, T, num_heads, head_dim//2]
    """
    因为拆分的是dim维度，所以直接一分为二后即可，不需要在意是奇数维还是偶数维
    同时，位置信息也是被注入到dim维度中（只需要在点乘时保证相对位置关系即可）
    """
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    x_rotated = x_rotated.to(x.dtype)  # 保持一致的dtype和shape
    return x_rotated


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_q_head = config.n_q_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_q_head
        assert self.n_embd % self.n_q_head == 0
        assert self.n_kv_head <= self.n_q_head and self.n_q_head % self.n_kv_head == 0
        self.q_proj = nn.Linear(self.n_embd, self.n_q_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # 计算QKV矩阵
        q = self.q_proj(x).view(B, T, self.n_q_head, self.head_dim)  # (B, T, n_q_head, head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)  # (B, T, n_kv_head, head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)  # (B, T, n_kv_head, head_dim)

        # 旋转位置编码（应用于Q和K）
        if cos_sin:
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)  # (B, T, n_q_head, head_dim)
            k = apply_rotary_emb(k, cos, sin)  # (B, T, n_kv_head, head_dim)

        # q, k 归一化
        q = norm(q)
        k = norm(k)

        # 维度变换，方便后续计算
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # KV缓存机制
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        # 分类讨论
        T_q = q.size(2)  # 查询序列长度
        T_k = k.size(2)  # 键值序列长度

        # situation 1: T_q == T_k 或者 无缓存
        # - 训练阶段 or 推理阶段的第一个token（prefill阶段）
        if T_q == T_k or kv_cache is None:
            # 直接用 F.scaled_dot_product_attention
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                is_causal=True,
                enable_gqa=self.n_q_head == self.n_kv_head,
            )
        # situation 2: T_k == 1 且 有缓存
        # - 推理阶段的后续token（decode阶段）
        elif T_k == 1:
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                is_causal=False,
                enable_gqa=self.n_q_head == self.n_kv_head,
            )
        # situation 3:
        # - 推理阶段中出现chunk query输入（感觉这种情况可能会比较少，在需要控制模型输出的时候会用到）
        else:
            attn_mask = torch.zeros((T_q, T_k), dtype=torch.bool, device=x.device)
            prefix_len = T_k - T_q
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True  # 屏蔽掉prefix部分

            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((T_q, T_q), dtype=torch.bool, device=x.device)
            ) == 0  # 下三角为False，上三角为True
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                enable_gqa=self.n_q_head == self.n_kv_head,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(attn_output)
        return y


class MLP(nn.Module):
    """
    MLP: 简单的升降维+relu^2激活, 即先 ReLU 再平方（比常规 ReLU 更稀疏）
    (一种比较落后的MLP层，现在MLP通常都有gate了)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.up_proj = nn.Linear(self.n_embd, 4 * self.n_embd, bias=False)
        self.down_proj = nn.Linear(4 * self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x).square()
        x = self.down_proj(x)
        return x


class Layer(nn.Module):
    """
    GPT每一层的结构：自注意力子层 + MLP子层 + 残差连接 + RMSNorm
    """

    def __init__(self, config: GPTConfig, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # 自注意力子层
        attn_output = self.attn(norm(x), cos_sin, kv_cache)
        x = x + attn_output  # 残差连接

        # MLP子层
        mlp_output = self.mlp(norm(x))
        x = x + mlp_output  # 残差连接

        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            Layer(config, layer_idx) for layer_idx in range(config.n_layer)
        ])

        # 语言模型头，将隐藏状态映射到词表
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 旋转位置编码相关参数，暂时先初始化一个较长的序列长度
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_q_head
        cos, sin = self._precompute_rotary_embeddings(
            seq_len=self.rotary_seq_len,
            head_dim=head_dim,
        )

        # 注册为buffer，方便后续调用（非参数但需要持续使用）
        # persistent=False 表示这些buffer不参与模型的state_dict保存和加载
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.cos, self.sin = cos, sin

    # 预计算旋转位置编码的cos和sin矩阵
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # 预计算旋转位置编码的cos和sin矩阵
        if device is None:
            device = self.get_device()
        # 位置编码的倒数频率
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 位置索引
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # add batch and head dims for later broadcasting
        return cos, sin

    # === 一些基础功能 ===
    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """ 估计每生成一个token所需的FLOPs
        参考: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_q_head, self.config.n_embd // self.config.n_q_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    # 前向计算
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.config.sequence_len, "Cannot forward, model sequence length is exhausted."

        pos0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, pos0:pos0 + T, :, :], self.sin[:, pos0:pos0 + T, :, :]

        x = self.wte(idx)  # token embedding
        for layer in self.layers:
            x = layer(x, cos_sin, kv_cache)
        x = norm(x)  # final RMSNorm

        # lm_head 计算 logits 和 loss
        softcap = 15
        if targets:
            # training mode
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction=loss_reduction,
            )
            return logits, loss
        else:
            # inference mode
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            return logits


