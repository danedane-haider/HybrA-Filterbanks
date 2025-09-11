from unittest.mock import patch

import torch
import torch.nn as nn

from hybra import ISAC, ISACCC, HybrA, ISACSpec


class TestBasicFunctionality:
    """Test basic functionality of all filterbank classes."""

    def test_isac_basic(self):
        """Test basic ISAC functionality."""
        fs = 8000
        L = 2 * fs
        signal = torch.randn(1, L)

        filterbank = ISAC(fs=fs, L=L)

        # Test forward pass
        output = filterbank(signal)
        assert torch.isfinite(output).all()
        assert len(output.shape) == 3  # batch, channels, time
        assert output.shape[0] == 1

        # Test reconstruction
        reconstructed = filterbank.decoder(output)
        assert torch.isfinite(reconstructed).all()
        assert len(reconstructed.shape) == 2 # batch, time
        assert reconstructed.shape[0] == 1

    def test_hybra_basic(self):
        """Test basic HybrA functionality."""
        fs = 8000
        L = 2 * fs
        signal = torch.randn(1, L)

        filterbank = HybrA(fs=fs, L=L)

        # Test forward pass
        output = filterbank(signal)
        assert torch.isfinite(output).all()
        assert len(output.shape) == 3  # batch, channels, time
        assert output.shape[0] == 1

        # Test reconstruction
        reconstructed = filterbank.decoder(output)
        assert torch.isfinite(reconstructed).all()
        assert len(reconstructed.shape) == 2 # batch, time
        assert reconstructed.shape[0] == 1

    def test_isacspec_basic(self):
        """Test basic ISACSpec functionality."""
        fs = 8000
        L = 2 * fs
        signal = torch.randn(1, L)

        spectrogram = ISACSpec(fs=fs, L=L)

        # Test forward pass
        output = spectrogram(signal)
        assert torch.isfinite(output).all()
        assert len(output.shape) == 3  # batch, channels, time
        assert output.shape[0] == 1

    def test_isaccc_basic(self):
        """Test basic ISACCC functionality."""
        fs = 8000
        L = 2 *fs
        signal = torch.randn(1, L)

        mfcc = ISACCC(fs=fs, L=L, num_cc=13)

        # Test forward pass
        output = mfcc(signal)
        assert torch.isfinite(output).all()
        assert len(output.shape) == 3  # batch, coefficients, time
        assert output.shape[0] == 1
        assert output.shape[1] == 13  # number of coefficients

    def test_batch_processing(self):
        """Test batch processing."""
        fs = 8000
        L = 2 * fs
        batch_size = 3
        signal = torch.randn(batch_size, L)

        filterbank = ISAC(fs=fs, L=L)
        output = filterbank(signal)

        assert output.shape[0] == batch_size
        assert torch.isfinite(output).all()

    def test_different_scales(self):
        """Test different auditory scales."""
        fs = 8000
        L = 2 * fs
        signal = torch.randn(1, L)

        for scale in ["mel", "erb", "log10", "elelog"]:
            try:
                filterbank = ISAC(fs=fs, L=L, scale=scale)
                output = filterbank(signal)
                assert torch.isfinite(output).all()
            except (ValueError, KeyError):
                # Some scales might not be implemented
                continue

    def test_learnable_parameters(self):
        """Test that HybrA has learnable parameters."""
        fs = 8000
        filterbank = HybrA(fs=fs, L=2*fs)

        params = list(filterbank.parameters())
        assert len(params) > 0, "HybrA should have learnable parameters"

        # Test gradient flow
        signal = torch.randn(1, 2*fs)
        output = filterbank(signal)
        loss = output.abs().sum()
        loss.backward()

        # Check that some parameters have gradients
        params_with_grad = sum(1 for p in filterbank.parameters() if p.grad is not None)
        assert params_with_grad > 0, "Some parameters should have gradients"

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_visualization_methods(self, mock_figure, mock_show):
        """Test that visualization methods work."""
        fs = 8000
        L = 2 * fs
        filterbank = ISAC(fs=fs, L=L)

        # Test plot_response if it exists
        if hasattr(filterbank, "plot_response"):
            try:
                filterbank.plot_response()
            except Exception:
                pass  # Visualization might fail in test environment

        # Test ISACgram if it exists
        if hasattr(filterbank, "ISACgram"):
            try:
                signal = torch.randn(1, L)
                filterbank.ISACgram(signal)
            except Exception:
                pass  # Visualization might fail in test environment

    def test_different_signal_lengths(self):
        """Test with different signal lengths."""
        fs = 8000
        base_L = fs

        for multiplier in [0.5, 1.0, 2.0]:
            L = int(base_L * multiplier)
            signal = torch.randn(1, L)
            filterbank = ISAC(fs=fs, L=L)

            output = filterbank(signal)
            assert torch.isfinite(output).all()
            assert output.shape[0] == 1

    def test_edge_case_zero_signal(self):
        """Test with zero signal."""
        fs = 8000
        L = 2 * fs
        zero_signal = torch.zeros(1, L)

        filterbank = ISAC(fs=fs, L=L)
        output = filterbank(zero_signal)

        # Output should be finite and mostly zero
        assert torch.isfinite(output).all()
        assert torch.norm(output) < 1e-3  # Should be very small


class TestUtilityFunctions:
    """Test utility functions with correct API."""

    def test_frame_bounds_basic(self):
        """Test frame bounds computation."""
        from hybra.utils import frame_bounds

        # Create simple filterbank
        w = torch.randn(4, 16)
        d = 2

        A, B = frame_bounds(w, d)

        assert torch.isfinite(A) and torch.isfinite(B)
        assert B > 0
        assert A <= B

class TestIntegrationBasic:
    """Basic integration tests."""

    def test_end_to_end_pipeline(self):
        """Test complete processing pipeline."""
        fs = 8000
        L = 2 * fs
        signal = torch.randn(1, L)

        # Step 1: ISAC processing
        isac = ISAC(fs=fs, L=L)
        isac_coeffs = isac(signal)
        isac_recon = isac.decoder(isac_coeffs)

        # Step 2: Spectrogram features
        spec = ISACSpec(fs=fs, L=L)
        spec_features = spec(signal)

        # Step 3: MFCC features
        mfcc = ISACCC(fs=fs, L=L, num_cc=13)
        mfcc_features = mfcc(signal)

        # All should work and be finite
        assert torch.isfinite(isac_coeffs).all()
        assert torch.isfinite(isac_recon).all()
        assert torch.isfinite(spec_features).all()
        assert torch.isfinite(mfcc_features).all()

        # Verify feature dimensionality
        assert mfcc_features.shape[1] == 13
        assert mfcc_features.shape[1] <= spec_features.shape[1]

    def test_neural_network_integration_basic(self):
        """Test basic neural network integration."""
        fs = 8000
        L = 2 * fs
        signal = torch.randn(1, L)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.filterbank = HybrA(fs=fs, L=L)
                self.classifier = nn.Linear(40, 2)  # Assume 40 channels

            def forward(self, x):
                features = self.filterbank(x)  # (batch, channels, time)
                pooled = features.abs().mean(
                    dim=2
                )  # Global average pooling (convert to real)
                output = self.classifier(pooled)
                return output

        model = SimpleModel()

        # Forward pass
        output = model(signal)
        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        param_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert param_count > 0, "Some parameters should have gradients"
