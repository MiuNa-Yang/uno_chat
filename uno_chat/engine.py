import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque

from uno_chat.model.gpt import GPT
from uno_chat.tokenizer.tokenizer_hf import HuggingFaceTokenizer


# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None


def use_calculator(expr):
    """Evaluate a math expression safely."""
    expr = expr.replace(",", "")
    if any([x not in "0123456789*+-/.() " for x in expr]):  # for now disallow non-numeric chars
        return None
    if "**" in expr:  # for now disallow power operator, could be very expensive
        return None
    return eval_with_timeout(expr)


# ----- KV Cache -----
class KVCache:
    """
    简易的KV缓存实现
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        对于一个给定的Transformer模型，初始化KV缓存
        需要知道如下信息才能够正确分配缓存空间
        :param batch_size:
        :param num_heads:
        :param seq_len:
        :param head_dim:
        :param num_layers:
        """

        self.kv_shape = (
            num_layers,
            2,  # key + value
            batch_size,
            num_heads,
            seq_len,
            head_dim
        )
        self.cache = None  # prefill阶段再分配
        self.pos = 0

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other: "KVCache"):

        # 检查兼容性
        assert self.cache is None, "Cannot prefill a non-empty KV cache"
        assert other.cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, batch_size, num_heads, head_dim must match
                assert dim1 == dim2, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) 初始化缓存
        dtype, device = other.cache.dtype, other.cache.device
        self.cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.cache[:, :, :, :, :other.pos, :] = other.cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache
        if self.cache is None:
            dtype, device = k.dtype, k.device
            self.cache = torch.empty(self.kv_shape, dtype=dtype, device=device)

        # 加入新的增量 key/value 对，并返回全量 key/value 对
        B, H, T, D = k.size()
        t0 = self.pos
        t1 = t0 + T
        # 动态扩容
        if t1 > self.cache.size(4):
            t_needed = t1 + 1024  # 增加一些余量，避免频繁扩容
            t_needed = (t_needed + 1023) & ~1023  # 向上对齐到1024的整数倍
            cur_shape = list(self.kv_shape)
            cur_shape[4] = t_needed
            self.cache.resize_(cur_shape)
        # 插入新的key/value
        self.cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.cache[layer_idx, 1, :, :, t0:t1, :] = v
        # 返回全量key/value
        full_k = self.cache[layer_idx, 0, :, :, :t1, :]
        full_v = self.cache[layer_idx, 1, :, :, :t1, :]
        # 如果是最后一层，更新pos
        if layer_idx == self.kv_shape[0] - 1:
            self.pos = t1
        return full_k, full_v


class SampleState:
    """
    采样状态
    """

    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()  # 需要强制输出的token队列
        self.in_python_block = False  # 是否在python代码块内
        self.python_expr_tokens = []  # 当前python代码块内的token列表
        self.completed = False  # 是否已经完成采样


class Engine:
    """
    简易的推理引擎
    """

    def __init__(self, model: GPT, tokenizer: HuggingFaceTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def sample_next_token(self, logits, rng, temperature=0.8, top_k=None):
        if temperature == 0.0:
            # 贪婪采样
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            return next_token
        if top_k is not None:
            # Top-K采样
            topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
            probs = F.softmax(topk_logits / temperature, dim=-1)
            next_token_in_topk = torch.multinomial(probs, num_samples=1, generator=rng)
            next_token = topk_indices.gather(-1, next_token_in_topk)
            return next_token

        # 温度采样
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=rng)
        return next_token

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        与model.generate一致，但是增加prefill+kv_cache的支持
        :param tokens:
        :param num_samples:
        :param max_tokens:
        :param temperature:
        :param top_k:
        :param seed:
        :return:
        """

        device = self.model.get_device()

        # 随机采样器（确保结果可以复现）
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 获取special tokens
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = get_special("<|bos|>")

        # Prompt Prefill阶段
        cfg = self.model.config
        # 与KV Cache相关的配置
        kv_kwargs = {
            "num_heads": cfg.n_kv_head,  # 一共多少个KV头
            "head_dim": cfg.n_embd // cfg.n_q_head,  # 每个头的维度
            "num_layers": cfg.n_layer,  # 一共多少层
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_kwargs
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = self.sample_next_token(logits, rng, temperature, top_k)
        sampled_tokens = next_ids[:, 0].tolist()

        # decode阶段(需要考虑多样本)
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_kwargs
        )
        # 相当于使用prefill的结果初始化decode的kv_cache
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # 释放内存

        # 初始化每个样本的状态
        sample_states = [SampleState(tokens.copy()) for _ in range(num_samples)]

        # 生成循环
        num_generated = 0  # 已经生成的token数量
        first_iteration = True
        while True:
            # 检查结束条件
            if max_tokens is not None and num_generated >= max_tokens:
                break

            if all(s.completed for s in sample_states):
                break

            if first_iteration:
                # 第一轮使用prefill的结果
                sampled_tokens = [sampled_tokens[0]] * num_samples  # Broadcast first token to all rows
                first_iteration = False
            else:
                # 后续轮次使用上一次采样的结果
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size)
                next_ids = self.sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_mask = []  # 需要mask掉python执行结果的token（不参与训练）
            for i, s in enumerate(sample_states):
                # 选择下一个token
                is_forced = len(s.forced_tokens) > 0
                token_mask.append(0 if is_forced else 1)
                next_token = s.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                s.current_tokens.append(next_token)

                # 处理特殊token逻辑
                # 1. assistant_end/bos: 结束采样
                if next_token in [assistant_end, bos]:
                    s.completed = True

                # 2. python_start: 进入python代码块
                elif next_token == python_start:
                    s.in_python_block = True
                    s.python_expr_tokens = []

                # 3. python_end: 执行python代码块
                elif next_token == python_end and s.in_python_block:
                    s.in_python_block = False
                    if s.python_expr_tokens:
                        expr = self.tokenizer.decode(
                            s.python_expr_tokens
                        )
                        result = use_calculator(expr)
                        if result is not None:
                            result_str = str(result)
                            result_tokens = self.tokenizer.encode(result_str)
                            s.forced_tokens.append(output_start)
                            s.forced_tokens.extend(result_tokens)
                            s.forced_tokens.append(output_end)
                    s.python_expr_tokens = []

                # 4. 其他token: 如果在python代码块内，收集代码
                elif s.in_python_block:
                    s.python_expr_tokens.append(next_token)

            yield token_column, token_mask
            num_generated += 1

            # 为下一轮生成准备输入
            ids = torch.tensor([token_column], dtype=torch.long, device=device).view(num_samples, 1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        批量生成接口
        :param tokens: List[List[int]]
        :param num_samples:
        :param kwargs:
        :return:
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.encode_special("<|bos|>")
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks