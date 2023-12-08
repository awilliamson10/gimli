from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer, AddedToken
import transformers

LLAMA_DEFAULT_EOS_TOKEN = "</s>"


@dataclass
class TokenizerConfig:
    use_fast: Optional[bool] = True
    tokenizer_type: Optional[str] = "LlamaTokenizer"
    tokenizer_model: Optional[str] = "NousResearch/Llama-2-7b-hf"
    trust_remote_code: Optional[bool] = False
    special_tokens: Optional[dict] = None
    tokens: Optional[list] = None


def load_tokenizer(cfg):
    tokenizer_kwargs = {}
    use_fast = True  # this is the default

    if cfg.use_fast is not None:
        use_fast = cfg.use_fast

    tokenizer_cls = AutoTokenizer
    if cfg.tokenizer_type:
        tokenizer_cls = getattr(transformers, cfg.tokenizer_type)

    tokenizer = tokenizer_cls.from_pretrained(
        cfg.tokenizer_model,
        trust_remote_code=cfg.trust_remote_code or False,
        use_fast=use_fast,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__
        in [
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
        ]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # set a pad_token, but use eos_token so we don't add a new token
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    if cfg.special_tokens:
        for k, val in cfg.special_tokens.items():
            tokenizer.add_special_tokens(
                {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
            )
    if cfg.tokens:
        tokenizer.add_tokens(
            [
                AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                for token in cfg.tokens
            ]
        )

    return tokenizer
