#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data loaders for evaluate_ppl_chunked.py

Provides multiple loading modes:
- jsonl_text: Load raw text from JSONL files (current behavior)
- amaia: Load from AMAIA's build_datasets pipeline with packing/masking
- tokens_file: Load pre-tokenized sequences from NPZ or JSONL files
"""

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np


@dataclass
class Sample:
    """
    A tokenized sample with optional mask for loss computation.

    Attributes
    ----------
    tokens : List[int]
        Token IDs for the sequence
    mask : Optional[List[bool]]
        If provided, mask[i] = True means token i should be included in loss.
        Follows AMAIA convention: mask[i+1] applies to predicting token i+1 from token i.
        If None, all tokens are included in loss computation.
    src : Optional[Any]
        Optional source metadata (e.g., file path, dataset name)
    """
    tokens: List[int]
    mask: Optional[List[bool]] = None
    src: Optional[Any] = None

    def __post_init__(self):
        """Validate mask length matches tokens if provided."""
        if self.mask is not None and len(self.mask) != len(self.tokens):
            raise ValueError(
                f"Mask length ({len(self.mask)}) must match tokens length ({len(self.tokens)})"
            )

    def validate(self) -> None:
        """Validate the sample data."""
        if not self.tokens:
            raise ValueError("Sample must have at least one token")
        if self.mask is not None and len(self.mask) != len(self.tokens):
            raise ValueError(
                f"Mask length ({len(self.mask)}) must match tokens length ({len(self.tokens)})"
            )

    def to_serializable(self) -> Dict:
        """Convert to a dict for multi-GPU pickling."""
        return {
            "tokens": self.tokens,
            "mask": self.mask,
            "src": self.src if isinstance(self.src, (str, int, float, type(None))) else str(self.src),
        }

    @classmethod
    def from_serializable(cls, data: Dict) -> "Sample":
        """Reconstruct a Sample from a serialized dict."""
        return cls(
            tokens=data["tokens"],
            mask=data.get("mask"),
            src=data.get("src"),
        )

    def __len__(self) -> int:
        return len(self.tokens)


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> List[Sample]:
        """Load and return a list of Sample objects."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of the loader and its configuration."""
        pass


class JSONLTextLoader(BaseLoader):
    """
    Load raw text from JSONL files and tokenize on-the-fly.

    This is the original behavior of evaluate_ppl_chunked.py.
    Each line in the JSONL file should have a text field (auto-detected).
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_samples: Optional[int] = None,
        max_length: Optional[int] = None,
        text_field: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        file_path : str
            Path to the JSONL file
        tokenizer : PreTrainedTokenizer
            HuggingFace tokenizer for encoding text
        max_samples : int, optional
            Maximum number of samples to load
        max_length : int, optional
            Maximum sequence length (truncate longer sequences)
        text_field : str, optional
            Field name containing text. If None, auto-detects from "text", "content",
            or uses the first string field.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_length = max_length
        self.text_field = text_field

    def _detect_text_field(self, record: Dict) -> str:
        """Auto-detect the text field from a record."""
        for field in ["text", "content"]:
            if field in record and isinstance(record[field], str):
                return field
        for key, value in record.items():
            if isinstance(value, str):
                return key
        raise ValueError(f"Could not detect text field in record with keys: {list(record.keys())}")

    def load(self) -> List[Sample]:
        """Load JSONL file and tokenize each text."""
        samples = []
        detected_field = None

        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if self.max_samples is not None and i >= self.max_samples:
                    break

                record = json.loads(line)

                if detected_field is None:
                    detected_field = self.text_field or self._detect_text_field(record)

                text = record[detected_field]
                tokens = self.tokenizer.encode(text)

                if self.max_length is not None and len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]

                samples.append(Sample(
                    tokens=tokens,
                    mask=None,  # All tokens count toward loss
                    src=self.file_path,
                ))

        return samples

    def get_description(self) -> str:
        return f"JSONLTextLoader(file={self.file_path}, max_samples={self.max_samples})"


class TokensFileLoader(BaseLoader):
    """
    Load pre-tokenized sequences from NPZ or JSONL files.

    Supports:
    - NPZ files: Arrays named "tokens" and optionally "mask"
    - JSONL files: Each line has "tokens" and optionally "mask" fields
    """

    def __init__(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        file_format: str = "auto",
        token_field: str = "tokens",
        mask_field: str = "mask",
    ):
        """
        Parameters
        ----------
        file_path : str
            Path to the token file (.npz or .jsonl)
        max_samples : int, optional
            Maximum number of samples to load
        file_format : str
            File format: "auto", "npz", or "jsonl"
        token_field : str
            Field name for tokens (default: "tokens")
        mask_field : str
            Field name for mask (default: "mask")
        """
        self.file_path = file_path
        self.max_samples = max_samples
        self.file_format = file_format
        self.token_field = token_field
        self.mask_field = mask_field

    def _detect_format(self) -> str:
        """Auto-detect file format from extension."""
        path = Path(self.file_path)
        if path.suffix == ".npz":
            return "npz"
        elif path.suffix in [".jsonl", ".json"]:
            return "jsonl"
        else:
            raise ValueError(f"Cannot auto-detect format for {path.suffix}. Use file_format='npz' or 'jsonl'")

    def _load_npz(self) -> List[Sample]:
        """Load samples from NPZ file."""
        data = np.load(self.file_path, allow_pickle=True)
        tokens_arr = data[self.token_field]
        mask_arr = data.get(self.mask_field)

        samples = []
        n_samples = len(tokens_arr)
        if self.max_samples is not None:
            n_samples = min(n_samples, self.max_samples)

        for i in range(n_samples):
            tokens = tokens_arr[i].tolist() if hasattr(tokens_arr[i], 'tolist') else list(tokens_arr[i])
            mask = None
            if mask_arr is not None:
                mask = mask_arr[i].tolist() if hasattr(mask_arr[i], 'tolist') else list(mask_arr[i])
                mask = [bool(m) for m in mask]

            samples.append(Sample(tokens=tokens, mask=mask, src=self.file_path))

        return samples

    def _load_jsonl(self) -> List[Sample]:
        """Load samples from JSONL file."""
        samples = []

        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if self.max_samples is not None and i >= self.max_samples:
                    break

                record = json.loads(line)
                tokens = record[self.token_field]
                mask = record.get(self.mask_field)

                if mask is not None:
                    mask = [bool(m) for m in mask]

                samples.append(Sample(tokens=tokens, mask=mask, src=self.file_path))

        return samples

    def load(self) -> List[Sample]:
        """Load pre-tokenized samples from file."""
        fmt = self.file_format if self.file_format != "auto" else self._detect_format()

        if fmt == "npz":
            return self._load_npz()
        elif fmt == "jsonl":
            return self._load_jsonl()
        else:
            raise ValueError(f"Unknown format: {fmt}")

    def get_description(self) -> str:
        return f"TokensFileLoader(file={self.file_path}, format={self.file_format}, max_samples={self.max_samples})"


class AMAIALoader(BaseLoader):
    """
    Load data using AMAIA's build_datasets pipeline.

    This loader lazily imports AMAIA modules and uses the same data processing
    pipeline as AMAIA's perplexity evaluation, including tokenization and packing/masking.
    """

    def __init__(
        self,
        sources_config: str,
        seq_len: int = 2048,
        max_samples: Optional[int] = None,
        seed: int = 42,
        shuffle_buffer_size: int = 1,
        tokenizer_name: str = "instruct_tiktoken_v6",
        tokenizer_path: Optional[str] = None,
        amaia_path: str = "/storage/home/manufay/amaia",
    ):
        """
        Parameters
        ----------
        sources_config : str
            Either a path to a YAML/JSON config file defining sources, or a comma-separated
            list of data paths (treated as pretrain type).
        seq_len : int
            Sequence length for packing
        max_samples : int, optional
            Maximum number of samples to load
        seed : int
            Random seed for shuffling
        shuffle_buffer_size : int
            Shuffle buffer size (1 = no shuffling)
        tokenizer_name : str
            AMAIA tokenizer name (e.g., "instruct_tiktoken_v6", "bytes")
        tokenizer_path : str, optional
            Path to tokenizer model file (required for some tokenizers)
        amaia_path : str
            Path to AMAIA repository for imports
        """
        self.sources_config = sources_config
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
        self.amaia_path = amaia_path

    def _parse_sources_config(self) -> List[Dict]:
        """Parse sources config into list of source dicts."""
        config_path = Path(self.sources_config)

        if config_path.exists() and config_path.suffix in [".yaml", ".yml", ".json"]:
            # Load from config file
            import yaml
            with open(config_path, 'r') as f:
                if config_path.suffix == ".json":
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)

            if isinstance(config, dict) and "sources" in config:
                return config["sources"]
            elif isinstance(config, list):
                return config
            else:
                raise ValueError(f"Invalid sources config format in {config_path}")
        else:
            # Treat as comma-separated paths
            paths = [p.strip() for p in self.sources_config.split(",") if p.strip()]
            return [{"path": p, "type": "pretrain"} for p in paths]

    def load(self) -> List[Sample]:
        """Load samples using AMAIA's build_datasets pipeline."""
        # Lazy import AMAIA
        if self.amaia_path not in sys.path:
            sys.path.insert(0, self.amaia_path)

        try:
            from amaia.text.tokenizers import build_tokenizer
            from apps.llm.utils.data_builder import DataSourceArgs, build_datasets
        except ImportError as e:
            raise ImportError(
                f"Failed to import AMAIA modules from {self.amaia_path}. "
                f"Ensure AMAIA is installed and the path is correct. Error: {e}"
            )

        # Build tokenizer
        tokenizer = build_tokenizer(self.tokenizer_name, self.tokenizer_path)

        # Parse sources config
        sources_dicts = self._parse_sources_config()
        sources = [
            DataSourceArgs(
                path=s["path"],
                type=s.get("type", "pretrain"),
                weight=s.get("weight", 1.0),
            )
            for s in sources_dicts
        ]

        # Build datasets (single process, no distributed)
        datasets, _, _ = build_datasets(
            tokenizer=tokenizer,
            seq_len=self.seq_len,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_buffer_allow_flush=False,
            seed=self.seed,
            sources=sources,
            dp_world_size=1,
            dp_rank=0,
        )

        # Collect samples from all datasets
        samples = []
        samples_per_dataset = (
            self.max_samples // len(datasets) if self.max_samples else None
        )

        for dataset, source_dict in zip(datasets, sources_dicts):
            count = 0
            for datum in dataset:
                tokens = datum.val.tolist() if hasattr(datum.val, 'tolist') else list(datum.val)
                mask = None
                if datum.mask is not None:
                    mask = datum.mask.tolist() if hasattr(datum.mask, 'tolist') else list(datum.mask)
                    mask = [bool(m) for m in mask]

                samples.append(Sample(
                    tokens=tokens,
                    mask=mask,
                    src=source_dict.get("path", "amaia"),
                ))

                count += 1
                if samples_per_dataset and count >= samples_per_dataset:
                    break
                if self.max_samples and len(samples) >= self.max_samples:
                    break

            if self.max_samples and len(samples) >= self.max_samples:
                break

        return samples[:self.max_samples] if self.max_samples else samples

    def get_description(self) -> str:
        return f"AMAIALoader(sources={self.sources_config}, seq_len={self.seq_len}, max_samples={self.max_samples})"


def create_loader(
    loader_type: str,
    *,
    # Common args
    max_samples: Optional[int] = None,
    # JSONLTextLoader args
    file_path: Optional[str] = None,
    tokenizer=None,
    max_length: Optional[int] = None,
    text_field: Optional[str] = None,
    # TokensFileLoader args
    token_data_path: Optional[str] = None,
    token_format: str = "auto",
    token_field: str = "tokens",
    mask_field: str = "mask",
    # AMAIALoader args
    amaia_sources_config: Optional[str] = None,
    amaia_seq_len: int = 2048,
    amaia_seed: int = 42,
    amaia_shuffle_buffer_size: int = 1,
    amaia_tokenizer_name: str = "instruct_tiktoken_v6",
    amaia_tokenizer_path: Optional[str] = None,
    amaia_path: str = "/storage/home/manufay/amaia",
) -> BaseLoader:
    """
    Factory function to create a data loader.

    Parameters
    ----------
    loader_type : str
        Type of loader: "jsonl_text", "amaia", or "tokens_file"
    max_samples : int, optional
        Maximum number of samples to load

    For jsonl_text loader:
    - file_path: Path to JSONL file
    - tokenizer: HuggingFace tokenizer
    - max_length: Maximum sequence length
    - text_field: Field name containing text

    For tokens_file loader:
    - token_data_path: Path to token file (.npz or .jsonl)
    - token_format: "auto", "npz", or "jsonl"
    - token_field: Field name for tokens
    - mask_field: Field name for mask

    For amaia loader:
    - amaia_sources_config: Path to config or comma-separated data paths
    - amaia_seq_len: Sequence length
    - amaia_seed: Random seed
    - amaia_shuffle_buffer_size: Shuffle buffer size
    - amaia_tokenizer_name: AMAIA tokenizer name
    - amaia_tokenizer_path: Path to tokenizer model
    - amaia_path: Path to AMAIA repository

    Returns
    -------
    BaseLoader
        The configured loader instance
    """
    if loader_type == "jsonl_text":
        if file_path is None:
            raise ValueError("file_path is required for jsonl_text loader")
        if tokenizer is None:
            raise ValueError("tokenizer is required for jsonl_text loader")
        return JSONLTextLoader(
            file_path=file_path,
            tokenizer=tokenizer,
            max_samples=max_samples,
            max_length=max_length,
            text_field=text_field,
        )

    elif loader_type == "tokens_file":
        path = token_data_path or file_path
        if path is None:
            raise ValueError("token_data_path (or file_path) is required for tokens_file loader")
        return TokensFileLoader(
            file_path=path,
            max_samples=max_samples,
            file_format=token_format,
            token_field=token_field,
            mask_field=mask_field,
        )

    elif loader_type == "amaia":
        if amaia_sources_config is None:
            raise ValueError("amaia_sources_config is required for amaia loader")
        return AMAIALoader(
            sources_config=amaia_sources_config,
            seq_len=amaia_seq_len,
            max_samples=max_samples,
            seed=amaia_seed,
            shuffle_buffer_size=amaia_shuffle_buffer_size,
            tokenizer_name=amaia_tokenizer_name,
            tokenizer_path=amaia_tokenizer_path,
            amaia_path=amaia_path,
        )

    else:
        raise ValueError(f"Unknown loader type: {loader_type}. Choose from: jsonl_text, amaia, tokens_file")
