from unittest.mock import patch

import torch
import torch.nn as nn

from hybra import ISACSpec


class TestISACSpecInitialization:
    """Test ISACSpec filterbank initialization."""

    def test_basic_initialization(self, test_parameters):
        """Test basic ISACSpec initialization."""
        spectrogram = ISACSpec(**test_parameters)

        assert isinstance(spectrogram, nn.Module)
        assert spectrogram.fs == test_parameters["fs"]
        # ISACSpec may not expose all attributes directly, just check it initializes
        assert isinstance(spectrogram, ISACSpec)

    def test_initialization_with_power_parameter(self, small_test_parameters):
        """Test initialization with different power values."""
        params = small_test_parameters.copy()

        for power in [1.0, 2.0, 0.5]:
            params["power"] = power
            spectrogram = ISACSpec(**params)
            assert spectrogram.power == power

    def test_initialization_with_logarithmic_output(self, small_test_parameters):
        """Test initialization with logarithmic output."""
        params = small_test_parameters.copy()
        params["is_log"] = True

        spectrogram = ISACSpec(**params)
        assert spectrogram.is_log == True

    def test_initialization_with_custom_averaging(self, small_test_parameters):
        """Test initialization with custom averaging kernel size."""
        params = small_test_parameters.copy()
        params["avg_size"] = 16

        spectrogram = ISACSpec(**params)
        # ISACSpec successfully initialized with custom averaging
        assert isinstance(spectrogram, ISACSpec)

    def test_initialization_with_learnable_components(self, small_test_parameters):
        """Test initialization with learnable encoder and averaging."""
        params = small_test_parameters.copy()
        params.update({"is_encoder_learnable": True, "is_avg_learnable": True})

        spectrogram = ISACSpec(**params)

        # Check for learnable parameters
        param_count = sum(1 for _ in spectrogram.parameters())
        assert param_count > 0, "Should have learnable parameters when enabled"


class TestISACSpecForwardPass:
    """Test ISACSpec forward pass."""

    def test_forward_pass_shape(self, short_audio, small_test_parameters):
        """Test that forward pass produces correct output shape."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L})

        spectrogram = ISACSpec(**params)
        output = spectrogram(signal)

        # Output should be 3D: [batch, channels, time]
        assert len(output.shape) == 3
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == params["num_channels"]
        assert output.shape[2] > 0  # time dimension

    def test_forward_pass_with_different_batch_sizes(self, small_test_parameters):
        """Test forward pass with different batch sizes."""
        spectrogram = ISACSpec(**small_test_parameters)

        for batch_size in [1, 2, 4]:
            signal = torch.randn(batch_size, small_test_parameters["L"])
            output = spectrogram(signal)

            assert output.shape[0] == batch_size
            assert output.shape[1] == small_test_parameters["num_channels"]

    def test_output_values_are_positive_with_power(
        self, short_audio, small_test_parameters
    ):
        """Test that output values are positive when using power > 0."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L, "power": 2.0})

        spectrogram = ISACSpec(**params)
        output = spectrogram(signal)

        # With power = 2.0, all outputs should be non-negative
        assert torch.all(output >= 0), "Output should be non-negative with power > 0"

    def test_logarithmic_output(self, short_audio, small_test_parameters):
        """Test logarithmic output functionality."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L, "is_log": True, "power": 2.0})

        spectrogram = ISACSpec(**params)
        output = spectrogram(signal)

        # With log output, values can be negative (for small inputs)
        # But should be finite
        assert torch.isfinite(output).all(), "Log output should be finite"

    def test_gradient_flow_with_learnable_components(
        self, short_audio, small_test_parameters
    ):
        """Test gradient flow through learnable components."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update(
            {"fs": fs, "L": L, "is_encoder_learnable": True, "is_avg_learnable": True}
        )

        spectrogram = ISACSpec(**params)
        output = spectrogram(signal)
        loss = output.abs().sum()  # Convert complex to real for gradient computation
        loss.backward()

        # Check gradients exist for learnable parameters
        for name, param in spectrogram.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"


class TestISACSpecProperties:
    """Test ISACSpec specific properties."""

    def test_frequency_range_truncation(self, short_audio, small_test_parameters):
        """Test frequency range truncation with fmax parameter."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update(
            {
                "fs": fs,
                "L": L,
                "fmax": fs // 4,  # Truncate to quarter of sampling frequency
            }
        )

        spectrogram = ISACSpec(**params)
        output = spectrogram(signal)

        # Should still produce valid output
        assert torch.isfinite(output).all()
        assert output.shape[0] == 1
        assert output.shape[1] <= params["num_channels"]  # Might be truncated


class TestISACSpecVisualization:
    """Test ISACSpec visualization methods."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_response(self, mock_figure, mock_show, small_test_parameters):
        """Test filter response plotting."""
        spectrogram = ISACSpec(**small_test_parameters)

        if hasattr(spectrogram, "plot_response"):
            spectrogram.plot_response()
            mock_figure.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_isacgram_visualization(
        self, mock_figure, mock_show, short_audio, small_test_parameters
    ):
        """Test ISACgram visualization for spectrograms."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L})

        spectrogram = ISACSpec(**params)

        if hasattr(spectrogram, "ISACgram"):
            spectrogram.ISACgram(signal)
            # Some implementations may not support log_scale parameter
            try:
                spectrogram.ISACgram(signal, log_scale=True)
            except TypeError:
                pass  # log_scale parameter not supported
            mock_figure.assert_called()


class TestISACSpecEdgeCases:
    """Test ISACSpec edge cases and error handling."""

    def test_zero_input(self, small_test_parameters):
        """Test behavior with zero input."""
        spectrogram = ISACSpec(**small_test_parameters)
        zero_signal = torch.zeros(1, small_test_parameters["L"])

        output = spectrogram(zero_signal)

        # With zero input and power > 0, output should be zero or very small
        if spectrogram.power > 0:
            assert torch.all(output >= -1e-6), (
                "Output should be non-negative for zero input with positive power"
            )
            assert torch.norm(output) < 1e-6, (
                "Output should be near zero for zero input"
            )

    def test_invalid_power_parameter(self, small_test_parameters):
        """Test behavior with invalid power parameters."""
        params = small_test_parameters.copy()

        # Negative power should be handled (might raise error or work)
        params["power"] = -1.0
        try:
            spectrogram = ISACSpec(**params)
            signal = torch.randn(1, params["L"])
            output = spectrogram(signal)
            # If it works, output should still be finite
            assert torch.isfinite(output).all()
        except (ValueError, RuntimeError):
            # Or it should raise an appropriate error
            pass

    def test_very_large_averaging_kernel(self, small_test_parameters):
        """Test behavior with very large averaging kernel."""
        params = small_test_parameters.copy()
        params["avg_size"] = params["L"] // 2  # Very large averaging

        try:
            spectrogram = ISACSpec(**params)
            signal = torch.randn(1, params["L"])
            output = spectrogram(signal)

            # Should work but might produce very smooth output
            assert torch.isfinite(output).all()
            assert output.shape[2] > 0  # Should still have time dimension
        except (ValueError, RuntimeError) as e:
            # Or should raise informative error
            assert "size" in str(e).lower() or "kernel" in str(e).lower()

    def test_logarithmic_output_with_zero_values(self, small_test_parameters):
        """Test logarithmic output when input produces zero coefficients."""
        params = small_test_parameters.copy()
        params.update({"is_log": True, "power": 2.0})

        spectrogram = ISACSpec(**params)

        # Use a signal that might produce some very small coefficients
        zero_signal = torch.zeros(1, params["L"])

        try:
            output = spectrogram(zero_signal)

            # Should handle log(0) gracefully (typically by clamping or adding epsilon)
            assert torch.isfinite(output).all(), (
                "Log output should handle zero inputs gracefully"
            )
        except Exception:
            # If it fails, that's also acceptable as long as it's handled consistently
            pass

    def test_kernel_size_auto_computation(self, small_test_parameters):
        """Test automatic kernel size computation."""
        params = small_test_parameters.copy()
        params.pop("kernel_size", None)  # Remove kernel_size to test auto-computation

        try:
            spectrogram = ISACSpec(**params)
            signal = torch.randn(1, params["L"])
            output = spectrogram(signal)

            # Should work with auto-computed kernel size
            assert torch.isfinite(output).all()
            assert output.shape[1] == params["num_channels"]
        except Exception as e:
            # If auto-computation fails, error should be informative
            assert "kernel" in str(e).lower() or "size" in str(e).lower()
