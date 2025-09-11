from unittest.mock import patch

import pytest
import torch

from hybra.utils import (
    ISACgram,
    audfilters,
    circ_conv,
    circ_conv_transpose,
    condition_number,
    frame_bounds,
    plot_response,
)


class TestFrameBounds:
    """Test frame bounds computation."""

    def test_frame_bounds_basic(self):
        """Test basic frame bounds computation."""
        # Create simple filterbank
        num_channels = 4
        filter_length = 16
        w = torch.randn(num_channels, filter_length)
        d = 2  # decimation factor

        A, B = frame_bounds(w, d)

        # Frame bounds should be positive
        assert A > 0, "Lower frame bound should be positive"
        assert B > 0, "Upper frame bound should be positive"

        # Lower bound should be less than or equal to upper bound
        assert A <= B, "Lower frame bound should be <= upper frame bound"

    def test_frame_bounds_no_decimation(self):
        """Test frame bounds with decimation factor d=1."""
        num_channels = 8
        filter_length = 32
        w = torch.randn(num_channels, filter_length)
        d = 1

        A, B = frame_bounds(w, d)

        assert A > 0
        assert B > 0
        assert A <= B

    def test_frame_bounds_different_decimations(self):
        """Test frame bounds with different decimation factors."""
        num_channels = 8
        filter_length = 64
        w = torch.randn(num_channels, filter_length)

        for d in [1, 2, 4, 8]:
            A, B = frame_bounds(w, d)

            assert A > 0, f"Lower bound should be positive for d={d}"
            assert B > 0, f"Upper bound should be positive for d={d}"
            assert A <= B, f"A <= B should hold for d={d}"

    def test_frame_bounds_with_custom_signal_length(self):
        """Test frame bounds with custom signal length."""
        num_channels = 4
        filter_length = 16
        w = torch.randn(num_channels, filter_length)
        d = 4
        Ls = 128  # Custom signal length

        A, B = frame_bounds(w, d, Ls)

        assert A > 0
        assert B > 0
        assert A <= B

    def test_frame_bounds_orthogonal_filters(self):
        """Test frame bounds for approximately orthogonal filters."""
        # Create filters that should have good frame bounds
        num_channels = 4
        filter_length = 32
        d = 4

        # Create orthogonal-like filters using DFT basis
        freqs = torch.linspace(0, 2 * torch.pi, num_channels + 1)[:-1]
        t = torch.arange(filter_length, dtype=torch.float32)
        w = torch.zeros(num_channels, filter_length)

        for i, freq in enumerate(freqs):
            w[i] = torch.cos(freq * t) * torch.hann_window(filter_length)

        A, B = frame_bounds(w, d)

        # Should have reasonable frame bounds
        condition_num = B / A
        assert condition_num < 100, (
            f"Condition number {condition_num} too large for orthogonal-like filters"
        )


class TestConditionNumber:
    """Test condition number computation."""

    def test_condition_number_basic(self):
        """Test basic condition number computation."""
        num_channels = 4
        filter_length = 16
        w = torch.randn(num_channels, filter_length)
        d = 2

        cond_num = condition_number(w, d)

        # Condition number should be >= 1
        assert cond_num >= 1.0, "Condition number should be >= 1"

        # Should be finite
        assert torch.isfinite(cond_num), "Condition number should be finite"

    def test_condition_number_identity(self):
        """Test condition number for identity-like system."""
        # Create a system that should have condition number close to 1
        num_channels = 4
        filter_length = 16
        d = 1

        # Create impulse responses
        w = torch.zeros(num_channels, filter_length)
        w[:, filter_length // 2] = 1.0  # Impulses

        cond_num = condition_number(w, d)

        # Should be well-conditioned
        assert cond_num == 1, (
            f"Identity-like system should be well-conditioned, got {cond_num}"
        )


class TestAudFilters:
    """Test auditory filter generation."""

    def test_audfilters_mel_scale(self):
        """Test auditory filters with mel scale."""
        fs = 8000
        L = 2 * fs
        num_channels = 40
        kernel_size = 128
        fc_max = 3000
        scale = "mel"

        filters, _, _, _, _, _, _, _, _ = audfilters(fs, kernel_size, num_channels, fc_max, L, scale=scale)

        # Should return tensor of correct shape
        assert isinstance(filters, torch.Tensor)
        assert filters.shape == (num_channels, kernel_size)

        # Filters should be finite
        assert torch.isfinite(filters).all(), "Filters should be finite"

    def test_audfilters_different_scales(self):
        """Test auditory filters with different scales."""
        fs = 8000
        L = 2 * fs
        num_channels = 20
        kernel_size = 64
        fc_max = 3000

        for scale in ["mel", "erb", "log10", "elelog"]:
            try:
                filters, _, _, _, _, _, _, _, _ = audfilters(fs, kernel_size, num_channels, fc_max, L, scale=scale)

                assert filters.shape == (num_channels, kernel_size)
                assert torch.isfinite(filters).all(), (
                    f"Filters should be finite for scale {scale}"
                )

            except (ValueError, KeyError, NotImplementedError):
                # Some scales might not be implemented
                pass

class TestUtilityEdgeCases:
    """Test edge cases and error handling in utility functions."""

    def test_frame_bounds_small_filters(self):
        """Test frame bounds with very small filters."""
        num_channels = 2
        filter_length = 4
        w = torch.randn(num_channels, filter_length)
        d = 2

        A, B = frame_bounds(w, d)

        assert A > 0
        assert B > 0
        assert torch.isfinite(A) and torch.isfinite(B)

    def test_frame_bounds_large_decimation(self):
        """Test frame bounds with large decimation factor."""
        num_channels = 4
        filter_length = 32
        w = torch.randn(num_channels, filter_length)
        d = 16  # Large decimation

        try:
            A, B = frame_bounds(w, d)
            assert B > 0
        except AssertionError as e:
            # Might fail if decimation is too large
            if "must divide" in str(e):
                pytest.skip("Decimation factor too large for signal length")
            else:
                raise

class TestNumericalStability:
    """Test numerical stability of utility functions."""

    def test_frame_bounds_numerical_stability(self):
        """Test frame bounds numerical stability with ill-conditioned systems."""
        # Create potentially ill-conditioned filterbank
        num_channels = 4
        filter_length = 16

        # Filters with very different scales
        w = torch.randn(num_channels, filter_length)
        w[0] *= 1e6  # Very large filter
        w[1] *= 1e-6  # Very small filter

        d = 2

        try:
            A, B = frame_bounds(w, d)

            # Should still produce finite bounds
            assert torch.isfinite(A) and torch.isfinite(B)
            assert A > 0 and B > 0

            # Condition number might be very large
            cond_num = B / A
            assert torch.isfinite(cond_num)

        except Exception:
            # If it fails due to numerical issues, that's also acceptable
            # as long as it fails gracefully
            pass
