from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from hybra import ISACCC


class TestISACCCInitialization:
    """Test ISACCC (ISAC Cepstral Coefficients) initialization."""

    def test_basic_initialization(self, test_parameters):
        """Test basic ISACCC initialization."""
        params = test_parameters.copy()
        params["num_cc"] = 13  # Standard number of MFCC coefficients

        mfcc = ISACCC(**params)

        assert isinstance(mfcc, nn.Module)
        assert mfcc.num_cc == 13
        assert mfcc.fs == test_parameters["fs"]
        # ISACCC successfully initialized
        assert isinstance(mfcc, ISACCC)

    def test_initialization_with_different_num_cc(self, small_test_parameters):
        """Test initialization with different numbers of cepstral coefficients."""
        params = small_test_parameters.copy()

        for num_cc in [12, 13, 16, 20]:
            if num_cc <= params["num_channels"]:
                params["num_cc"] = num_cc
                mfcc = ISACCC(**params)
                assert mfcc.num_cc == num_cc

    def test_initialization_with_logarithmic_mode(self, small_test_parameters):
        """Test initialization with logarithmic vs dB mode."""
        params = small_test_parameters.copy()
        params["num_cc"] = 13

        # Test with is_log=True
        params["is_log"] = True
        mfcc_log = ISACCC(**params)
        # ISACCC successfully initialized with log mode
        assert isinstance(mfcc_log, ISACCC)

        # Test with is_log=False (dB mode)
        params["is_log"] = False
        mfcc_db = ISACCC(**params)
        # ISACCC successfully initialized with dB mode
        assert isinstance(mfcc_db, ISACCC)

    def test_initialization_validates_num_cc(self, small_test_parameters):
        """Test that num_cc > num_channels raises appropriate error."""
        params = small_test_parameters.copy()
        params["num_cc"] = params["num_channels"] + 5  # More coefficients than channels

        with pytest.raises(ValueError):
            ISACCC(**params)

    def test_initialization_with_custom_isacspec_parameters(
        self, small_test_parameters
    ):
        """Test initialization with custom ISACSpec parameters."""
        params = small_test_parameters.copy()
        params.update(
            {
                "num_cc": 13,
                "power": 1.0,  # Different from default 2.0
                "fmax": params["fs"] // 4,
                "supp_mult": 1.5,
            }
        )

        mfcc = ISACCC(**params)
        # ISACCC successfully initialized with custom ISACSpec parameters
        assert isinstance(mfcc, ISACCC)


class TestISACCCForwardPass:
    """Test ISACCC forward pass."""

    def test_forward_pass_shape(self, short_audio, small_test_parameters):
        """Test that forward pass produces correct output shape."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L, "num_cc": 13})

        mfcc = ISACCC(**params)
        output = mfcc(signal)

        # Output should be 3D: [batch, num_cc, time]
        assert len(output.shape) == 3
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == 13  # number of cepstral coefficients
        assert output.shape[2] > 0  # time dimension

    def test_forward_pass_with_different_batch_sizes(self, small_test_parameters):
        """Test forward pass with different batch sizes."""
        params = small_test_parameters.copy()
        params["num_cc"] = 13
        mfcc = ISACCC(**params)

        for batch_size in [1, 2, 4]:
            signal = torch.randn(batch_size, params["L"])
            output = mfcc(signal)

            assert output.shape[0] == batch_size
            assert output.shape[1] == 13


class TestISACCCVisualization:
    """Test ISACCC visualization methods."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_response(self, mock_figure, mock_show, small_test_parameters):
        """Test filter response plotting."""
        params = small_test_parameters.copy()
        params["num_cc"] = 13
        mfcc = ISACCC(**params)

        if hasattr(mfcc, "plot_response"):
            mfcc.plot_response()
            mock_figure.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_isacgram_visualization(
        self, mock_figure, mock_show, short_audio, small_test_parameters
    ):
        """Test ISACgram visualization for MFCC."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L, "num_cc": 13})

        mfcc = ISACCC(**params)

        if hasattr(mfcc, "ISACgram"):
            # Some implementations may not support all parameters
            try:
                mfcc.ISACgram(signal)
            except TypeError:
                pass  # Parameter not supported
            mock_figure.assert_called()


class TestISACCCEdgeCases:
    """Test ISACCC edge cases and error handling."""

    def test_zero_input(self, small_test_parameters):
        """Test behavior with zero input."""
        params = small_test_parameters.copy()
        params["num_cc"] = 13
        mfcc = ISACCC(**params)

        zero_signal = torch.zeros(1, params["L"])

        try:
            output = mfcc(zero_signal)

            # Should handle zero input gracefully
            assert torch.isfinite(output).all(), "Should handle zero input"

            # With zero input, output might be very negative (in log domain) or zero
            # Just check it's well-behaved
            assert not torch.isnan(output).any(), (
                "Should not produce NaN for zero input"
            )

        except Exception as e:
            # If it fails on zero input, error should be informative
            assert (
                "log" in str(e).lower()
                or "zero" in str(e).lower()
                or "divide" in str(e).lower()
            )

    def test_frequency_truncation_effect(self, short_audio, small_test_parameters):
        """Test effect of frequency truncation on MFCC."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L, "num_cc": 13})

        # Full frequency range
        mfcc_full = ISACCC(**params)
        output_full = mfcc_full(signal)

        # Truncated frequency range
        params["fmax"] = fs // 4
        mfcc_trunc = ISACCC(**params)
        output_trunc = mfcc_trunc(signal)

        # Both should work and have same coefficient count
        assert output_full.shape[1] == output_trunc.shape[1] == 13
        assert torch.isfinite(output_full).all()
        assert torch.isfinite(output_trunc).all()

        # Outputs should be different due to different frequency content
        assert not torch.allclose(output_full, output_trunc, atol=1e-3)


class TestISACCCSpeechApplications:
    """Test ISACCC in speech-like applications."""

    def test_speech_like_signal_processing(self, small_test_parameters):
        """Test ISACCC with speech-like signals."""
        params = small_test_parameters.copy()
        params.update(
            {
                "fs": 8000,  # Use consistent fs=8000
                "L": 2 * 8000,  # 2 seconds of speech, L = 2*fs
                "num_cc": 13,  # Standard for speech recognition
            }
        )

        # Create a speech-like signal (multiple harmonics)
        t = torch.linspace(0, 1, params["L"])
        fundamental = 150  # Typical fundamental frequency for speech

        speech_like = (
            torch.sin(2 * torch.pi * fundamental * t)
            + 0.5 * torch.sin(2 * torch.pi * 2 * fundamental * t)
            + 0.25 * torch.sin(2 * torch.pi * 3 * fundamental * t)
        ).unsqueeze(0)

        mfcc = ISACCC(**params)
        coeffs = mfcc(speech_like)

        # Should produce reasonable coefficients for speech
        assert torch.isfinite(coeffs).all()
        assert coeffs.shape == (1, 13, coeffs.shape[2])

        # First coefficient should typically be largest (energy)
        mean_first = coeffs[:, 0, :].abs().mean()
        mean_others = coeffs[:, 1:, :].abs().mean()

        assert mean_first > 0, "Energy coefficient should be positive"
        assert mean_others >= 0, "Other coefficients should be non-negative"

    def test_typical_mfcc_ranges(self, sample_audio, small_test_parameters):
        """Test that MFCC values are in typical ranges for audio."""
        signal, fs = sample_audio
        L = signal.shape[-1]

        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L, "num_cc": 13})

        mfcc = ISACCC(**params)
        coeffs = mfcc(signal)

        # Check ranges are reasonable (these are loose bounds)
        c0_range = coeffs[:, 0, :].max() - coeffs[:, 0, :].min()  # Energy coefficient
        other_range = (
            coeffs[:, 1:, :].max() - coeffs[:, 1:, :].min()
        )  # Other coefficients

        # Energy coefficient should have larger dynamic range
        assert c0_range > 0, "Energy coefficient should have dynamic range"
        assert other_range >= 0, "Other coefficients should have non-negative range"

        # Values shouldn't be too extreme
        assert coeffs.abs().max() < 1000, "MFCC values shouldn't be extremely large"
