import os
from pathlib import Path

from transformers import AutoTokenizer

# Determine offline/local behavior from env; default to allowing downloads
WANDB_MODE = os.environ.get("WANDB_MODE", "").lower()
HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
TRANSFORMERS_OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in {"1", "true", "yes"}
_default_local_only = WANDB_MODE == "offline" or HF_OFFLINE or TRANSFORMERS_OFFLINE

# Known local paths
_LOCAL_TOKENIZER_PATHS = {
    # SmolLM tokenizer cloned locally
    "smollm": "./hf_local/SmolLM-135M",
}


def _load_tokenizer_by_name(name: str):
    """Lazily construct a tokenizer by name without triggering other downloads."""
    if name == "smollm":
        # Force local for smollm
        return AutoTokenizer.from_pretrained(_LOCAL_TOKENIZER_PATHS["smollm"], local_files_only=True)

    # Fallback examples for other names (kept for compatibility). These will only run if requested.
    if name == "smollm2":
        # If a local clone exists, prefer it; else respect offline toggle
        local_clone = Path("./hf_local/SmolLM2-135M")
        if local_clone.exists():
            return AutoTokenizer.from_pretrained(str(local_clone), local_files_only=True)
        return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", local_files_only=_default_local_only)

    raise KeyError(f"Unknown tokenizer '{name}'. Add it to _load_tokenizer_by_name.")


class _TokenizerRegistry:
    """Dict-like lazy registry returning actual tokenizer instances on access."""

    def __init__(self):
        self._cache = {}

    def __getitem__(self, key: str):
        if key not in self._cache:
            self._cache[key] = _load_tokenizer_by_name(key)
        return self._cache[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        # Only advertise keys we know how to load lazily
        return {"smollm", "smollm2"}


# Public registry used across the codebase
TOKENIZERS = _TokenizerRegistry()


def load_tokenizer_from_config(cfg):
    name = getattr(cfg, "tokenizer", None)
    if not name:
        raise ValueError("Config missing 'tokenizer' field")

    tokenizer = TOKENIZERS[name]

    # Ensure a valid pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # As a last resort, add a dedicated pad token
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Optionally respect padding side if provided by cfg
    padding_side = getattr(cfg, "padding_side", None)
    if padding_side in ("left", "right"):
        tokenizer.padding_side = padding_side

    return tokenizer