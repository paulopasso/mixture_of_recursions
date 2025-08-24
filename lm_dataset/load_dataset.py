import os
import json
import warnings

import torch
from datasets import load_dataset, interleave_datasets

from lm_dataset.language_modeling_dataset import LanguageModelingDataset
from lm_dataset.tokenized_dataset import TokenizedCorpusDataset
from lm_dataset.data_preprocessing import AddLabels, RemoveIndex
from paths import DATA_DIR

num_proc = 24

# arguments for the load_dataset function
LM_DATASETS = {
    "slimpajama": {"path": f"{DATA_DIR}/slimpajama", "split": "train"},
    "slimpajama_chunk1": {"path": "json", "data_files": f"{DATA_DIR}/slimpajama_chunk1/*.jsonl", "split": "train"},
    "cosmopedia": {"path": f"{DATA_DIR}/cosmopedia-v2", "split": "train"},
    "fineweb_edu": {"path": f"{DATA_DIR}/fineweb-edu-dedup", "split": "train"},
    "fineweb_test": {"path": f"{DATA_DIR}/fineweb-test", "split": "train"},
    "python_edu": {"path": f"{DATA_DIR}/python-edu", "split": "train"},
    "open_web_math": {"path": f"{DATA_DIR}/open-web-math", "split": "train"}, 
    "math_code_pile": {"path": f"{DATA_DIR}/math-code-pile", "split": "train"}, 
    "starcoderdata": {"path": f"{DATA_DIR}/starcoderdata", "split": "train"},  # "data_dir": "python", 
    "finemath": {"path": f"{DATA_DIR}/finemath", "split": "train"},  # "name": "finemath-4plus", 
    # Small remote dataset for quick local/MPS debug
    "wikitext2": {"path": "wikitext", "name": "wikitext-2-raw-v1", "split": "train"},
}

# tokenizer used for pre-tokenization
TOKENIZED_DATASETS = {
    "pythia_pile": "pythia",  
}


def _is_local_json_source(path_str: str) -> bool:
    if os.path.isfile(path_str):
        return path_str.endswith(".json") or path_str.endswith(".jsonl")
    if os.path.isdir(path_str):
        # Check if directory contains json/jsonl files
        has_jsonl = any(f.endswith(".jsonl") for f in os.listdir(path_str))
        has_json = any(f.endswith(".json") for f in os.listdir(path_str))
        return has_jsonl or has_json
    return False


def _load_local_json_dataset(path_str: str):
    """Load a local JSON/JSONL file or directory as a streaming HF dataset."""
    if os.path.isfile(path_str):
        data_files = path_str
    elif os.path.isdir(path_str):
        # Prefer jsonl if present, else json
        if any(f.endswith(".jsonl") for f in os.listdir(path_str)):
            data_files = os.path.join(path_str, "*.jsonl")
        elif any(f.endswith(".json") for f in os.listdir(path_str)):
            data_files = os.path.join(path_str, "*.json")
        else:
            raise ValueError(f"No .json or .jsonl files found in directory: {path_str}")
    else:
        raise ValueError(f"Path does not exist: {path_str}")

    ds = load_dataset("json", data_files=data_files, split="train", streaming=True)
    return ds


def _ensure_text_column(ds, cfg):
    """Ensure dataset has a 'text' column by renaming if necessary."""
    # For streaming datasets, we can't easily check columns, so we'll try a different approach
    # First, check if user specified a text field
    text_field = getattr(cfg, "dataset_text_field", None)
    if text_field and text_field != "text":
        try:
            return ds.rename_column(text_field, "text")
        except Exception as e:
            print(f"Warning: Could not rename column '{text_field}' to 'text': {e}")
            return ds
    
    # If no text field specified, try to peek at the data to see what columns exist
    try:
        # Take a small sample to check columns
        sample = next(iter(ds.take(1)))
        available_columns = set(sample.keys())
        
        if "text" in available_columns:
            # Already has text column, no renaming needed
            return ds
        
        # Check for common alternatives and rename the first one found
        for alt in ["content", "document", "raw_text", "input", "data"]:
            if alt in available_columns:
                print(f"Renaming column '{alt}' to 'text'")
                return ds.rename_column(alt, "text")
        
        # If we get here, warn about missing text column
        print(f"Warning: No suitable text column found. Available columns: {available_columns}")
        return ds
        
    except Exception as e:
        print(f"Warning: Could not check dataset columns: {e}")
        # As a fallback, assume the dataset is already correct
        return ds


def load_dataset_from_config(cfg, tokenizer):
    dataset_name = cfg.dataset.split(',')
    dataset_name = [ds.strip() for ds in dataset_name]
    if len(dataset_name) > 1:
        # Allow mixing known LM_DATASETS with local json paths
        assert all((ds in LM_DATASETS) or _is_local_json_source(ds) for ds in dataset_name), "Only LM datasets or local JSON paths can be combined"
        assert "weights" in cfg, "When combining datasets, weights must be provided"
        assert len(dataset_name) == len(cfg.weights.split(',')), "Number of weights must match number of datasets"
    
    if all(ds in LM_DATASETS for ds in dataset_name) or all((ds in LM_DATASETS) or _is_local_json_source(ds) for ds in dataset_name):
        dataset_type = "lm"
        
        train_dataset = []
        for ds_name in dataset_name:
            if ds_name in LM_DATASETS:
                _dataset = load_dataset(**LM_DATASETS[ds_name], streaming=True)
                if ds_name == "starcoderdata":
                    _dataset = _dataset.rename_column("content", "text")
            elif _is_local_json_source(ds_name):
                _dataset = _load_local_json_dataset(ds_name)
                _dataset = _ensure_text_column(_dataset, cfg)
            else:
                raise ValueError(f"Unsupported dataset specifier: {ds_name}")
            train_dataset.append(_dataset)
        
        if len(train_dataset) == 1:
            train_dataset = train_dataset[0]
        else:
            train_dataset = interleave_datasets(train_dataset, probabilities=cfg.weights.split(','), seed=42)
        
    elif all(ds in TOKENIZED_DATASETS for ds in dataset_name):
        dataset_type = "token"
        # check if tokenizer used by dataset is compatible with the one specified in config
        if "tokenizer" in cfg:
            tokenizer_used = TOKENIZED_DATASETS[cfg.dataset]
            if cfg.tokenizer != tokenizer_used:
                raise ValueError(f"Tokenizer {cfg.tokenizer} is not compatible with dataset {cfg.dataset}")

        # load corpus
        if cfg.dataset == "pythia_pile":
            from lm_dataset.tokenized_dataset import PythiaPileTokenizedCorpus
            corpus = PythiaPileTokenizedCorpus(os.path.join(DATA_DIR, "pythia_pile_idxmaps"))

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    
    transforms = [
        AddLabels(),
        RemoveIndex(),
    ]
    
    if dataset_type == "lm":
        return LanguageModelingDataset(train_dataset, tokenizer, 
                                       max_length=cfg.max_length,
                                       transforms=transforms, 
                                       global_shuffling=cfg.get("global_shuffling", False),
                                       local_shuffling=cfg.get("local_shuffling", False),
                                       add_bos_token=cfg.get("add_bos_token", False),
                                       max_tokens=cfg.get("data_token_budget", None),
                                       max_batches=cfg.get("data_batch_budget", None),)
    
    elif dataset_type == "token":
        if cfg.dataloader_num_workers <= 1:
            warnings.warn(f"Using cfg.dataloader_num_workers={cfg.dataloader_num_workers} with TokenizedCorpusDataset."
                          f"You may want to increase this number to speed up data loading.")
        return TokenizedCorpusDataset(corpus, length=cfg.max_length, eos_token=tokenizer.eos_token_id,
                                      add_bos_token=cfg.get("add_bos_token", False),
                                      bos_token=tokenizer.bos_token_id, transforms=transforms,)    
      
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")