import numpy as np
import pytest
import torch


@pytest.fixture
def sample_audio():
    """Generate a sample audio signal for testing."""
    fs = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(fs * duration))

    # Create a signal with multiple frequency components
    signal = (
        0.5 * torch.sin(2 * torch.pi * 440 * t)  # A4 note
        + 0.3 * torch.sin(2 * torch.pi * 880 * t)  # A5 note
        + 0.2 * torch.sin(2 * torch.pi * 1320 * t)  # E6 note
    )

    return signal.unsqueeze(0), fs


@pytest.fixture
def short_audio():
    """Generate a short audio signal for testing."""
    fs = 16000
    L = 2 *fs
    t = torch.linspace(0, L / fs, L)

    # Simple sine wave
    signal = torch.sin(2 * torch.pi * 440 * t)

    return signal.unsqueeze(0), fs, L


@pytest.fixture
def noise_audio():
    """Generate white noise for testing."""
    fs = 16000
    L = 2*fs
    signal = torch.randn(1, L) * 0.1

    return signal, fs, L


@pytest.fixture
def test_parameters():
    """Common test parameters for filterbanks."""
    return {
        "kernel_size": 128,
        "num_channels": 40,
        "fs": 16000,
        "L": 16000,
        "scale": "mel",
    }


@pytest.fixture
def small_test_parameters():
    """Smaller parameters."""
    return {
        "kernel_size": 64,
        "num_channels": 20,
        "fs": 8000,
        "L": 2048,
        "scale": "mel",
    }


@pytest.fixture(params=["mel", "erb", "log10", "elelog"])
def scale_parameter(request):
    """Parametrized fixture for testing different scales."""
    return request.param


@pytest.fixture
def tolerance():
    """Standard tolerance for numerical comparisons."""
    return 1e-5


@pytest.fixture
def loose_tolerance():
    """Looser tolerance for reconstruction tests."""
    return 1e-3


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(0)
    np.random.seed(0)
