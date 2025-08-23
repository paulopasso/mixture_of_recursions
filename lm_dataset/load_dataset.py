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


def load_dataset_from_config(cfg, tokenizer):
    dataset_name = cfg.dataset.split(',')
    dataset_name = [ds.strip() for ds in dataset_name]
    if len(dataset_name) > 1:
        assert all(ds in LM_DATASETS for ds in dataset_name), "Only LM datasets can be combined"
        assert "weights" in cfg, "When combining datasets, weights must be provided"
        assert len(dataset_name) == len(cfg.weights.split(',')), "Number of weights must match number of datasets"
    
    if all(ds in LM_DATASETS for ds in dataset_name):
        dataset_type = "lm"
        
        train_dataset = []
        for ds in dataset_name:
            _dataset = load_dataset(**LM_DATASETS[ds], streaming=True)
            if ds == "starcoderdata":
                _dataset.rename_column("content", "text")
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