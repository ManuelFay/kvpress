# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for kvzap loaders module.

Tests Sample class, JSONLTextLoader, TokensFileLoader, and create_loader factory.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from kvzap.loaders import (
    Sample,
    BaseLoader,
    JSONLTextLoader,
    TokensFileLoader,
    create_loader,
)


# ==============================================================================
# Test Data Paths
# ==============================================================================

MOCK_DATA_DIR = Path(__file__).parent / "mock_data"


# ==============================================================================
# Sample Class Tests
# ==============================================================================

class TestSample:
    """Tests for the Sample dataclass."""

    def test_sample_creation_basic(self):
        """Test basic Sample creation with just tokens."""
        tokens = [100, 200, 300, 400, 500]
        sample = Sample(tokens=tokens)

        assert sample.tokens == tokens
        assert sample.mask is None
        assert sample.src is None
        assert len(sample) == 5

    def test_sample_creation_with_mask(self):
        """Test Sample creation with mask."""
        tokens = [100, 200, 300]
        mask = [True, False, True]
        sample = Sample(tokens=tokens, mask=mask)

        assert sample.tokens == tokens
        assert sample.mask == mask

    def test_sample_creation_with_src(self):
        """Test Sample creation with source metadata."""
        tokens = [100, 200]
        sample = Sample(tokens=tokens, src="test_file.jsonl")

        assert sample.src == "test_file.jsonl"

    def test_sample_mask_length_validation(self):
        """Test that mask length must match tokens length."""
        tokens = [100, 200, 300]
        mask = [True, False]  # Wrong length

        with pytest.raises(ValueError, match="Mask length.*must match tokens length"):
            Sample(tokens=tokens, mask=mask)

    def test_sample_validate_empty_tokens(self):
        """Test validation rejects empty tokens."""
        sample = Sample(tokens=[])

        with pytest.raises(ValueError, match="must have at least one token"):
            sample.validate()

    def test_sample_validate_valid(self):
        """Test validation passes for valid sample."""
        sample = Sample(tokens=[100, 200, 300])
        sample.validate()  # Should not raise

    def test_sample_to_serializable(self):
        """Test conversion to serializable dict."""
        tokens = [100, 200, 300]
        mask = [True, True, False]
        sample = Sample(tokens=tokens, mask=mask, src="test.jsonl")

        serialized = sample.to_serializable()

        assert serialized["tokens"] == tokens
        assert serialized["mask"] == mask
        assert serialized["src"] == "test.jsonl"

    def test_sample_from_serializable(self):
        """Test reconstruction from serialized dict."""
        data = {
            "tokens": [100, 200, 300],
            "mask": [True, True, False],
            "src": "test.jsonl"
        }

        sample = Sample.from_serializable(data)

        assert sample.tokens == data["tokens"]
        assert sample.mask == data["mask"]
        assert sample.src == data["src"]

    def test_sample_roundtrip_serialization(self):
        """Test that serialize/deserialize roundtrip preserves data."""
        original = Sample(
            tokens=[100, 200, 300, 400],
            mask=[True, False, True, True],
            src="roundtrip_test"
        )

        serialized = original.to_serializable()
        reconstructed = Sample.from_serializable(serialized)

        assert reconstructed.tokens == original.tokens
        assert reconstructed.mask == original.mask
        assert reconstructed.src == original.src

    def test_sample_len(self):
        """Test __len__ method."""
        sample = Sample(tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert len(sample) == 10


# ==============================================================================
# TokensFileLoader Tests
# ==============================================================================

class TestTokensFileLoader:
    """Tests for TokensFileLoader."""

    def test_load_jsonl_tokens(self):
        """Test loading pre-tokenized data from JSONL."""
        loader = TokensFileLoader(
            file_path=str(MOCK_DATA_DIR / "sample_tokens.jsonl"),
            max_samples=None,
        )

        samples = loader.load()

        assert len(samples) == 3
        assert samples[0].tokens == [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        assert samples[0].mask == [True] * 10
        assert samples[1].mask[0] is False  # First element is False
        assert samples[2].mask is None  # No mask in third line

    def test_load_jsonl_with_max_samples(self):
        """Test loading with max_samples limit."""
        loader = TokensFileLoader(
            file_path=str(MOCK_DATA_DIR / "sample_tokens.jsonl"),
            max_samples=2,
        )

        samples = loader.load()

        assert len(samples) == 2

    def test_load_npz_tokens(self):
        """Test loading from NPZ file."""
        # Create temporary NPZ file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tokens = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            mask = np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
            np.savez(f.name, tokens=tokens, mask=mask)
            temp_path = f.name

        try:
            loader = TokensFileLoader(file_path=temp_path, file_format="npz")
            samples = loader.load()

            assert len(samples) == 2
            assert samples[0].tokens == [1, 2, 3, 4, 5]
            assert samples[0].mask == [True, True, True, True, True]
            assert samples[1].mask[0] is False
        finally:
            os.unlink(temp_path)

    def test_format_auto_detection_jsonl(self):
        """Test auto-detection of JSONL format."""
        loader = TokensFileLoader(
            file_path=str(MOCK_DATA_DIR / "sample_tokens.jsonl"),
            file_format="auto",
        )

        samples = loader.load()
        assert len(samples) > 0

    def test_get_description(self):
        """Test description string generation."""
        loader = TokensFileLoader(
            file_path="/path/to/tokens.jsonl",
            max_samples=100,
        )

        desc = loader.get_description()

        assert "TokensFileLoader" in desc
        assert "tokens.jsonl" in desc
        assert "100" in desc

    def test_custom_field_names(self):
        """Test loading with custom field names."""
        # Create temp file with custom field names
        with tempfile.NamedTemporaryFile(mode='w', suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 0]}) + "\n")
            temp_path = f.name

        try:
            loader = TokensFileLoader(
                file_path=temp_path,
                token_field="input_ids",
                mask_field="attention_mask",
            )
            samples = loader.load()

            assert len(samples) == 1
            assert samples[0].tokens == [1, 2, 3]
            assert samples[0].mask == [True, True, False]
        finally:
            os.unlink(temp_path)


# ==============================================================================
# JSONLTextLoader Tests
# ==============================================================================

class TestJSONLTextLoader:
    """Tests for JSONLTextLoader."""

    def test_load_text_with_mock_tokenizer(self):
        """Test loading and tokenizing text."""
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [100, 200, 300, 400, 500]

        loader = JSONLTextLoader(
            file_path=str(MOCK_DATA_DIR / "sample_text.jsonl"),
            tokenizer=mock_tokenizer,
            max_samples=2,
        )

        samples = loader.load()

        assert len(samples) == 2
        assert mock_tokenizer.encode.call_count == 2
        assert samples[0].tokens == [100, 200, 300, 400, 500]
        assert samples[0].mask is None  # Text loader doesn't set masks

    def test_load_with_max_length(self):
        """Test truncation with max_length."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens

        loader = JSONLTextLoader(
            file_path=str(MOCK_DATA_DIR / "sample_text.jsonl"),
            tokenizer=mock_tokenizer,
            max_samples=1,
            max_length=50,
        )

        samples = loader.load()

        assert len(samples[0].tokens) == 50

    def test_auto_detect_text_field(self):
        """Test auto-detection of text field."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        loader = JSONLTextLoader(
            file_path=str(MOCK_DATA_DIR / "sample_text.jsonl"),
            tokenizer=mock_tokenizer,
            max_samples=1,
        )

        samples = loader.load()

        # Should have detected 'text' field and encoded it
        assert len(samples) == 1
        mock_tokenizer.encode.assert_called()

    def test_custom_text_field(self):
        """Test using custom text field."""
        # Create temp file with custom field
        with tempfile.NamedTemporaryFile(mode='w', suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"content": "Hello world"}) + "\n")
            temp_path = f.name

        try:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2]

            loader = JSONLTextLoader(
                file_path=temp_path,
                tokenizer=mock_tokenizer,
                text_field="content",
            )

            samples = loader.load()

            assert len(samples) == 1
            mock_tokenizer.encode.assert_called_with("Hello world")
        finally:
            os.unlink(temp_path)

    def test_get_description(self):
        """Test description string."""
        mock_tokenizer = MagicMock()

        loader = JSONLTextLoader(
            file_path="/path/to/data.jsonl",
            tokenizer=mock_tokenizer,
            max_samples=50,
        )

        desc = loader.get_description()

        assert "JSONLTextLoader" in desc
        assert "data.jsonl" in desc


# ==============================================================================
# create_loader Factory Tests
# ==============================================================================

class TestCreateLoader:
    """Tests for the create_loader factory function."""

    def test_create_tokens_file_loader(self):
        """Test creating TokensFileLoader via factory."""
        loader = create_loader(
            loader_type="tokens_file",
            token_data_path=str(MOCK_DATA_DIR / "sample_tokens.jsonl"),
            max_samples=2,
        )

        assert isinstance(loader, TokensFileLoader)
        samples = loader.load()
        assert len(samples) == 2

    def test_create_jsonl_text_loader(self):
        """Test creating JSONLTextLoader via factory."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        loader = create_loader(
            loader_type="jsonl_text",
            file_path=str(MOCK_DATA_DIR / "sample_text.jsonl"),
            tokenizer=mock_tokenizer,
            max_samples=1,
        )

        assert isinstance(loader, JSONLTextLoader)

    def test_create_loader_missing_args_tokens_file(self):
        """Test error when required args missing for tokens_file."""
        with pytest.raises(ValueError, match="token_data_path.*required"):
            create_loader(loader_type="tokens_file")

    def test_create_loader_missing_args_jsonl_text(self):
        """Test error when required args missing for jsonl_text."""
        with pytest.raises(ValueError, match="file_path is required"):
            create_loader(loader_type="jsonl_text")

    def test_create_loader_missing_tokenizer(self):
        """Test error when tokenizer missing for jsonl_text."""
        with pytest.raises(ValueError, match="tokenizer is required"):
            create_loader(
                loader_type="jsonl_text",
                file_path="/path/to/file.jsonl",
            )

    def test_create_loader_unknown_type(self):
        """Test error for unknown loader type."""
        with pytest.raises(ValueError, match="Unknown loader type"):
            create_loader(loader_type="unknown_loader")

    def test_create_loader_amaia_missing_config(self):
        """Test error when amaia config missing."""
        with pytest.raises(ValueError, match="amaia_sources_config is required"):
            create_loader(loader_type="amaia")


# ==============================================================================
# BaseLoader Abstract Tests
# ==============================================================================

class TestBaseLoader:
    """Tests for BaseLoader abstract class."""

    def test_cannot_instantiate_base_loader(self):
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()

    def test_subclass_must_implement_methods(self):
        """Test that subclass must implement abstract methods."""
        class IncompleteLoader(BaseLoader):
            pass

        with pytest.raises(TypeError):
            IncompleteLoader()
