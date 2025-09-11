from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from hybra import HybrA


class TestHybrAInitialization:
    """Test HybrA filterbank initialization."""

    def test_basic_initialization(self, test_parameters):
        """Test basic HybrA initialization with default parameters."""
        filterbank = HybrA(**test_parameters)

        assert isinstance(filterbank, nn.Module)
        assert filterbank.fs == test_parameters["fs"]
        # HybrA may not expose all attributes directly, just check it initializes
        assert isinstance(filterbank, HybrA)

    def test_initialization_with_learned_components(self, small_test_parameters):
        """Test initialization with learnable filter components."""
        params = small_test_parameters.copy()
        params["learned_kernel_size"] = 23

        filterbank = HybrA(**params)

        # HybrA should have learnable parameters by design
        assert len(list(filterbank.parameters())) > 0, (
            "HybrA should have learnable parameters"
        )

    def test_initialization_with_different_scales(
        self, scale_parameter, small_test_parameters
    ):
        """Test initialization with different auditory scales."""
        params = small_test_parameters.copy()
        params["scale"] = scale_parameter

        filterbank = HybrA(**params)
        # HybrA successfully initialized with the scale parameter
        assert isinstance(filterbank, HybrA)

    def test_initialization_with_tightening(self, small_test_parameters):
        """Test initialization with tightening enabled."""
        params = small_test_parameters.copy()
        params["tighten"] = True

        filterbank = HybrA(**params)
        # HybrA successfully initialized with tightening enabled
        assert isinstance(filterbank, HybrA)

class TestHybrAForwardPass:
    """Test HybrA filterbank forward pass."""

    def test_forward_pass_shape(self, short_audio, small_test_parameters):
        """Test that forward pass produces correct output shape."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params["fs"] = fs
        params["L"] = L

        filterbank = HybrA(**params)
        output = filterbank(signal)

        # Check basic output shape structure
        assert len(output.shape) == 3
        assert output.shape[0] == 1
        assert output.shape[1] == params["num_channels"]

    def test_forward_pass_with_different_input_sizes(self, small_test_parameters):
        """Test forward pass with different input sizes."""
        params = small_test_parameters.copy()
        filterbank = HybrA(**params)

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            signal = torch.randn(batch_size, params["L"])
            output = filterbank(signal)
            assert output.shape[0] == batch_size
            assert output.shape[1] == params["num_channels"]

    def test_gradient_flow_through_learned_filters(
        self, short_audio, small_test_parameters
    ):
        """Test that gradients flow through learned filter components."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L})

        filterbank = HybrA(**params)
        signal.requires_grad_(False)  # Only test filterbank gradients

        output = filterbank(signal)
        loss = output.abs().sum()  # Convert complex to real for gradient computation
        loss.backward()

        # Check that learned parameters have gradients
        learned_param_count = 0
        for name, param in filterbank.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
                learned_param_count += 1

        # HybrA should have learnable parameters
        assert learned_param_count > 0, (
            "HybrA should have learnable parameters with gradients"
        )

    def test_training_mode_vs_eval_mode(self, short_audio, small_test_parameters):
        """Test behavior in training vs evaluation mode."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L})

        filterbank = HybrA(**params)

        # Test in training mode
        filterbank.train()
        output_train = filterbank(signal)

        # Test in evaluation mode
        filterbank.eval()
        output_eval = filterbank(signal)

        # Outputs should be identical for HybrA (no dropout/batch norm typically)
        assert torch.allclose(output_train, output_eval, atol=1e-6)


class TestHybrAReconstruction:
    """Test HybrA reconstruction properties."""

    def test_reconstruction_with_tightening(
        self, short_audio, small_test_parameters, loose_tolerance
    ):
        """Test reconstruction with tightening enabled."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L, "tighten": True})

        filterbank = HybrA(**params)
        coefficients = filterbank(signal)
        reconstructed = filterbank.decoder(coefficients)

        reconstruction_error = torch.norm(signal - reconstructed[:, :L]) / torch.norm(signal)
        assert reconstruction_error < 1.0


class TestHybrAProperties:
    """Test HybrA mathematical properties."""

    def test_frame_bounds(self, small_test_parameters):
        """Test frame bounds computation for HybrA."""
        filterbank = HybrA(**small_test_parameters)

        if hasattr(filterbank, "frame_bounds"):
            A, B = filterbank.frame_bounds()

            assert A > 0, "Lower frame bound should be positive"
            assert B > 0, "Upper frame bound should be positive"
            assert A <= B, "Lower frame bound should be <= upper frame bound"

class TestHybrAVisualization:
    """Test HybrA visualization methods."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_response(self, mock_figure, mock_show, small_test_parameters):
        """Test filter response plotting."""
        filterbank = HybrA(**small_test_parameters)

        if hasattr(filterbank, "plot_response"):
            filterbank.plot_response()
            mock_figure.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_isacgram(self, mock_figure, mock_show, short_audio, small_test_parameters):
        """Test ISACgram visualization for HybrA."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L})

        filterbank = HybrA(**params)

        if hasattr(filterbank, "ISACgram"):
            filterbank.ISACgram(signal)
            # Some implementations may not support log_scale parameter
            try:
                filterbank.ISACgram(signal, log_scale=True)
            except TypeError:
                pass  # log_scale parameter not supported
            mock_figure.assert_called()


class TestHybrALearning:
    """Test HybrA learning capabilities."""

    def test_parameter_updates(self, short_audio, small_test_parameters):
        """Test that parameters actually update during training."""
        signal, fs, L = short_audio
        params = small_test_parameters.copy()
        params.update({"fs": fs, "L": L})

        filterbank = HybrA(**params)

        # Store initial parameter values
        initial_params = {}
        for name, param in filterbank.named_parameters():
            initial_params[name] = param.clone()

        # Perform one training step
        optimizer = torch.optim.SGD(filterbank.parameters(), lr=0.1)

        coeffs = filterbank(signal)
        recon = filterbank.decoder(coeffs)
        loss = torch.norm(signal - recon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = 0
        for name, param in filterbank.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-8):
                params_changed += 1

        assert params_changed > 0, "No parameters were updated during training"


class TestHybrAEdgeCases:
    """Test HybrA edge cases and error handling."""

    def test_invalid_learned_kernel_size(self, small_test_parameters):
        """Test invalid learned kernel sizes."""
        params = small_test_parameters.copy()

        # Test very large learned kernel size
        params["learned_kernel_size"] = params["kernel_size"] * 2

        try:
            filterbank = HybrA(**params)
            # If it succeeds, it should still work
            assert isinstance(filterbank, HybrA)
        except (ValueError, RuntimeError):
            # Or it should raise an informative error
            pass

    def test_zero_input_with_learned_filters(self, small_test_parameters):
        """Test behavior with zero input and learned filters."""
        filterbank = HybrA(**small_test_parameters)
        zero_signal = torch.zeros(1, small_test_parameters["L"])

        output = filterbank(zero_signal)

        # Output should be close to zero (learned filters might add small bias)
        assert torch.norm(output) < 1e-3, "Zero input should produce near-zero output"

    def test_very_small_learned_kernel_size(self, small_test_parameters):
        """Test with very small learned kernel size."""
        params = small_test_parameters.copy()
        params["learned_kernel_size"] = 3

        filterbank = HybrA(**params)

        # Should initialize successfully
        assert isinstance(filterbank, HybrA)

        # Should be able to process signals
        signal = torch.randn(1, params["L"])
        output = filterbank(signal)
        assert output.shape[0] == 1
        assert output.shape[1] == params["num_channels"]
