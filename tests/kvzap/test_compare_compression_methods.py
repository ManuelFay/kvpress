# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for compare_compression_methods.py

Tests press factory functions, visualization helpers, DEFAULT_THRESHOLDS,
and the main comparison infrastructure.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

# Import the module under test
from kvzap.compare_compression_methods import (
    DEFAULT_THRESHOLDS,
    create_random_press,
    create_streaming_llm_press,
    create_expected_attention_press,
    create_observed_attention_press,
    create_h2o_press,
    create_kvzap_press,
    visualize_compression_heatmap,
)

from kvzap.loaders import Sample


# ==============================================================================
# Mock Data Paths
# ==============================================================================

MOCK_DATA_DIR = Path(__file__).parent / "mock_data"


# ==============================================================================
# DEFAULT_THRESHOLDS Tests
# ==============================================================================

class TestDefaultThresholds:
    """Tests for DEFAULT_THRESHOLDS configuration."""

    def test_default_thresholds_exist(self):
        """Test that all expected thresholds are defined."""
        expected_keys = [
            "kvzap",
            "expected_attention",
            "random",
            "random_validation",
            "observed_attention",
            "h2o",
        ]

        for key in expected_keys:
            assert key in DEFAULT_THRESHOLDS, f"Missing threshold for {key}"

    def test_threshold_types(self):
        """Test that all thresholds are numeric."""
        for key, value in DEFAULT_THRESHOLDS.items():
            assert isinstance(value, (int, float)), f"{key} threshold is not numeric"

    def test_kvzap_threshold_negative(self):
        """Test that KVzap threshold is negative (log-probabilities)."""
        assert DEFAULT_THRESHOLDS["kvzap"] < 0, "KVzap threshold should be negative"

    def test_random_validation_threshold(self):
        """Test that random_validation threshold is 1.0 (matches StreamingLLM)."""
        assert DEFAULT_THRESHOLDS["random_validation"] == 1.0

    def test_random_threshold_range(self):
        """Test random threshold is in [0, 1] range."""
        assert 0 <= DEFAULT_THRESHOLDS["random"] <= 1


# ==============================================================================
# Press Factory Function Tests
# ==============================================================================

class TestCreateRandomPress:
    """Tests for create_random_press factory function."""

    def test_create_random_press_basic(self):
        """Test basic random press creation."""
        press = create_random_press(threshold=0.5, sliding_window_size=128)

        # Should return a DMSPress wrapping RandomPress
        assert press is not None
        assert hasattr(press, "threshold")
        assert press.threshold == 0.5

    def test_create_random_press_with_seed(self):
        """Test random press with specific seed."""
        press = create_random_press(threshold=0.3, seed=42)
        assert press is not None

    def test_create_random_press_with_sink_tokens(self):
        """Test random press with sink tokens."""
        press = create_random_press(threshold=0.5, n_sink=8)
        assert press is not None


class TestCreateStreamingLLMPress:
    """Tests for create_streaming_llm_press factory function."""

    def test_create_streaming_llm_press_basic(self):
        """Test basic StreamingLLM press creation."""
        press = create_streaming_llm_press(n_sink=4, sliding_window_size=128)

        assert press is not None
        # StreamingLLM wraps with threshold=0.5
        assert hasattr(press, "threshold")
        assert press.threshold == 0.5

    def test_create_streaming_llm_press_custom_params(self):
        """Test with custom parameters."""
        press = create_streaming_llm_press(n_sink=8, sliding_window_size=256)
        assert press is not None


class TestCreateExpectedAttentionPress:
    """Tests for create_expected_attention_press factory function."""

    def test_create_expected_attention_press_basic(self):
        """Test basic ExpectedAttention press creation."""
        press = create_expected_attention_press(threshold=0.05, sliding_window_size=128)

        assert press is not None
        assert hasattr(press, "threshold")

    def test_create_expected_attention_press_custom_params(self):
        """Test with custom parameters."""
        press = create_expected_attention_press(
            threshold=0.1,
            sliding_window_size=256,
            n_future_positions=1024,
            n_sink=8,
            use_covariance=False,
            use_vnorm=False,
        )
        assert press is not None


class TestCreateObservedAttentionPress:
    """Tests for create_observed_attention_press factory function."""

    def test_create_observed_attention_press_basic(self):
        """Test basic ObservedAttention press creation."""
        press = create_observed_attention_press(threshold=0.0005, sliding_window_size=128)

        assert press is not None
        assert hasattr(press, "threshold")

    def test_create_observed_attention_press_custom_window(self):
        """Test with custom sliding window size."""
        press = create_observed_attention_press(threshold=0.001, sliding_window_size=256)
        assert press is not None


class TestCreateH2OPress:
    """Tests for create_h2o_press factory function."""

    def test_create_h2o_press_basic(self):
        """Test basic H2O press creation."""
        press = create_h2o_press(
            threshold=0.0005,
            sliding_window_size=128,
            local_window_size=512,
        )

        assert press is not None
        assert hasattr(press, "threshold")

    def test_create_h2o_press_custom_params(self):
        """Test with custom parameters."""
        press = create_h2o_press(
            threshold=0.001,
            sliding_window_size=256,
            local_window_size=1024,
        )
        assert press is not None


class TestCreateKVzapPress:
    """Tests for create_kvzap_press factory function."""

    def test_create_kvzap_press_basic(self):
        """Test basic KVzap press creation (without scorer model)."""
        # This test may require the actual kvpress installation
        # Skip if KVzapPress not available
        try:
            press = create_kvzap_press(threshold=-6.0, sliding_window_size=128)
            assert press is not None
        except Exception as e:
            # If KVzap models aren't available, skip
            pytest.skip(f"KVzap models not available: {e}")

    def test_create_kvzap_press_with_sink(self):
        """Test KVzap press with sink tokens."""
        try:
            press = create_kvzap_press(threshold=-7.0, n_sink=8)
            assert press is not None
        except Exception as e:
            pytest.skip(f"KVzap models not available: {e}")


# ==============================================================================
# Visualization Tests
# ==============================================================================

class TestVisualizeCompressionHeatmap:
    """Tests for visualize_compression_heatmap function."""

    def test_visualize_empty_scores(self):
        """Test handling of empty scores dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty scores should just print warning and return
            visualize_compression_heatmap(
                scores_dict={},
                threshold=0.5,
                method_name="Test",
                output_dir=Path(tmpdir),
            )
            # Should not create any files
            assert len(list(Path(tmpdir).glob("*.png"))) == 0

    def test_visualize_with_mock_scores(self):
        """Test visualization with mock score data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock binary decisions (batch=1, heads=4, seq_len=128)
            mock_decisions = torch.ones(1, 4, 128)
            # Evict some tokens (positions 10-20)
            mock_decisions[:, :, 10:20] = 0

            scores_dict = {
                0: mock_decisions,
                1: mock_decisions.clone(),
                2: mock_decisions.clone(),
                3: mock_decisions.clone(),
            }

            visualize_compression_heatmap(
                scores_dict=scores_dict,
                threshold=0.5,
                method_name="TestMethod",
                output_dir=Path(tmpdir),
                max_tokens=128,
                sliding_window_size=16,
                n_sink=4,
            )

            # Check that PNG file was created
            png_files = list(Path(tmpdir).glob("*.png"))
            assert len(png_files) == 1
            assert "TestMethod" in png_files[0].name

    def test_visualize_with_head_disagreement(self):
        """Test visualization when heads disagree on eviction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create decisions where heads disagree
            mock_decisions = torch.ones(1, 4, 64)
            # Head 0 and 1 evict position 30, heads 2 and 3 keep it
            mock_decisions[:, 0:2, 30] = 0

            scores_dict = {i: mock_decisions.clone() for i in range(4)}

            visualize_compression_heatmap(
                scores_dict=scores_dict,
                threshold=0.5,
                method_name="HeadDisagreement",
                output_dir=Path(tmpdir),
                max_tokens=64,
            )

            png_files = list(Path(tmpdir).glob("*.png"))
            assert len(png_files) == 1


# ==============================================================================
# Sample Integration Tests
# ==============================================================================

class TestSampleIntegration:
    """Integration tests between Sample class and comparison methods."""

    def test_sample_to_dict_for_multigpu(self):
        """Test that Sample serialization works for multi-GPU pickling."""
        sample = Sample(
            tokens=[100, 200, 300, 400, 500],
            mask=[True, True, False, True, True],
            src="test_source"
        )

        serialized = sample.to_serializable()

        # Should be JSON serializable
        json_str = json.dumps(serialized)
        assert json_str

        # Should be reconstructible
        recovered = Sample.from_serializable(serialized)
        assert recovered.tokens == sample.tokens
        assert recovered.mask == sample.mask

    def test_sample_with_large_tokens(self):
        """Test Sample with realistic token count."""
        # Simulate a 2048 token sequence
        tokens = list(range(2048))
        mask = [True] * 2048

        sample = Sample(tokens=tokens, mask=mask)

        assert len(sample) == 2048
        serialized = sample.to_serializable()
        assert len(serialized["tokens"]) == 2048


# ==============================================================================
# Press Configuration Tests
# ==============================================================================

class TestPressConfiguration:
    """Tests for press configuration and parameters."""

    def test_dms_sliding_window_import(self):
        """Test that DMS_SLIDING_WINDOW_SIZE is accessible."""
        from kvzap.evaluate_ppl_chunked import DMS_SLIDING_WINDOW_SIZE

        assert isinstance(DMS_SLIDING_WINDOW_SIZE, int)
        assert DMS_SLIDING_WINDOW_SIZE > 0

    def test_press_threshold_assignment(self):
        """Test that thresholds are correctly assigned to presses."""
        test_threshold = 0.42

        press = create_random_press(threshold=test_threshold)
        assert press.threshold == test_threshold

        press = create_streaming_llm_press()
        assert press.threshold == 0.5  # StreamingLLM always uses 0.5

    def test_sliding_window_propagation(self):
        """Test that sliding window size propagates to presses."""
        window_size = 256

        press = create_random_press(threshold=0.5, sliding_window_size=window_size)
        assert hasattr(press, 'sliding_window_size')
        assert press.sliding_window_size == window_size


# ==============================================================================
# Edge Cases and Error Handling Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_sample_single_token(self):
        """Test Sample with single token."""
        sample = Sample(tokens=[100])
        assert len(sample) == 1
        sample.validate()  # Should not raise

    def test_sample_mask_all_false(self):
        """Test Sample with all-false mask."""
        sample = Sample(
            tokens=[100, 200, 300],
            mask=[False, False, False]
        )
        assert sample.mask == [False, False, False]

    def test_visualization_special_characters_in_name(self):
        """Test that special characters in method name are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_decisions = torch.ones(1, 2, 32)
            scores_dict = {0: mock_decisions}

            # Method name with special characters
            visualize_compression_heatmap(
                scores_dict=scores_dict,
                threshold=0.5,
                method_name="Random(threshold=1.0)",
                output_dir=Path(tmpdir),
                max_tokens=32,
            )

            # Should create file with sanitized name
            png_files = list(Path(tmpdir).glob("*.png"))
            assert len(png_files) == 1
            # Check that parentheses and equals are replaced
            assert "(" not in png_files[0].name
            assert "=" not in png_files[0].name

    def test_tokens_file_loader_empty_file(self):
        """Test handling of empty tokens file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".jsonl", delete=False) as f:
            temp_path = f.name
            # Don't write anything - empty file

        try:
            from kvzap.loaders import TokensFileLoader
            loader = TokensFileLoader(file_path=temp_path)
            samples = loader.load()
            assert len(samples) == 0
        finally:
            os.unlink(temp_path)
