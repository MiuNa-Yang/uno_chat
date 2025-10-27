import os

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class HuggingFaceTokenizer:
    """
    一个轻量化的HuggingFace分词器包装器，提供基本的分词和解码功能。
    （使用HuggingFace的tokenizers库中的工具提升性能）
    """

    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer

    # === 加载 ===
    @classmethod
    def from_pretrained(cls, hf_path: str):
        """
        从HuggingFace预训练模型路径加载分词器。
        Args:
            hf_path: HuggingFace模型路径或名称
        """
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_file(cls, tokenizer_file: str):
        """
        从本地文件加载分词器。
        Args:
            tokenizer_file: 分词器文件路径，或者包含tokenizer.json的目录路径
        """
        if os.path.isfile(tokenizer_file):
            tokenizer = HFTokenizer.from_file(tokenizer_file)
        else:
            tokenizer_file = os.path.join(tokenizer_file, "tokenizer.json")
            tokenizer = HFTokenizer.from_file(tokenizer_file)
        return cls(tokenizer)

    # === 训练BPE ===
    @classmethod
    def train_bpe(cls, text_iterator, vocab_size: int, special_tokens=None):
        """
        使用BPE算法训练一个新的分词器。
        复用tokenizers库Tokenizer的训练功能。
        Args:
            text_iterator: 可迭代的文本数据源
            vocab_size: 词汇表大小
            special_tokens: 需要添加的特殊标记列表
        """
        SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,  # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        gpt4_split_regex = Regex(SPLIT_PATTERN)  # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,  # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special_tokens,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    # === tokenizer所需的基础方法 ===
    def get_vocab_size(self):
        """ 获取词汇表大小 """
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        """ 获取特殊标记的字典 """
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, token_id: int) -> str:
        """
        根据token id获取对应的token字符串
        Args:
            token_id: token的整数ID
        Returns:
            对应的token字符串
        """
        return self.tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> int:
        """
        根据token字符串获取对应的token id
        Args:
            token: token字符串
        Returns:
            对应的token整数ID
        """
        return self.tokenizer.token_to_id(token)

    def _encode_one(self, text: str) -> list[int]:
        """
        将输入字符串编码为token id列表

        Args:
            text: 输入字符串
        Returns:
            token id的整数列表
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def encode_string(self, text: str) -> list[int]:
        """
        将输入字符串编码为token id列表
        Args:
            text: 输入字符串
        Returns:
            token id的整数列表
        """
        return self.tokenizer.token_to_id(text)

    def encode(self, text: str | list[str], *args, **kwargs) -> list[int]:
        """
        将输入字符串编码为token id列表
        Args:
            text: 输入字符串
        Returns:
            token id的整数列表
        """
        if isinstance(text, str):
            return self._encode_one(text)

        elif isinstance(text, list):
            all_ids = []
            for t in text:
                ids = self._encode_one(t)
                all_ids.extend(ids)
            return all_ids

        else:
            raise ValueError(f"Unsupported input type for encode: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, token_ids: list[int]) -> str:
        """
        将token id列表解码为字符串
        Args:
            token_ids: token id的整数列表
        Returns:
            解码后的字符串
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def save_pretrained(self, save_directory: str):
        """
        将分词器保存到指定目录
        Args:
            save_directory: 目标保存目录
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        save_path = os.path.join(save_directory, "tokenizer.json")
        self.tokenizer.save(save_path)
