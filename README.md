![Logo](https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/HybrA.png?raw=true)

**Auditory-inspired filterbanks for deep learning**

Welcome to HybrA-Filterbanks, a PyTorch library providing state-of-the-art auditory-inspired filterbanks for audio processing and deep learning applications.

## Overview

This library contains the official implementations of:

* **ISAC** ([paper](https://arxiv.org/abs/2505.07709)): Invertible and Stable Auditory filterbank with Customizable kernels for ML integration
* **HybrA** ([paper](https://arxiv.org/abs/2408.17358)): Hybrid Auditory filterbank that extends ISAC with learnable filters
* **ISACSpec**: Spectrogram variant with temporal averaging for robust feature extraction  
* **ISACCC**: Cepstral coefficient extractor for speech recognition applications

## Key Features

✨ **PyTorch Integration**: All filterbanks are implemented as `nn.Module` for seamless integration into neural networks

🎯 **Auditory Modeling**: Based on human auditory perception principles (mel, ERB, bark scales)

⚡ **Fast Implementation**: Optimized using FFT-based circular convolution

🔧 **Flexible Configuration**: Customizable kernel sizes, frequency ranges, and scales

📊 **Frame Theory**: Built-in functions for frame bounds, condition numbers, and stability analysis

🎨 **Visualization**: Rich plotting capabilities for filter responses and time-frequency representations 

## Documentation
[https://github.com/danedane-haider/HybrA-Filterbanks](https://danedane-haider.github.io/HybrA-Filterbanks/main/)

## Installation
We publish all releases on PyPi. You can install the current version by running:
```
pip install hybra
```

## Quick Start

### Basic ISAC Filterbank

```python
import torch
import torchaudio
from hybra import ISAC

# Load audio signal
x, fs = torchaudio.load("your_audio.wav")
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
L = x.shape[-1]

# Create ISAC filterbank
isac_fb = ISAC(
    kernel_size=1024, 
    num_channels=128, 
    L=L, 
    fs=fs,
    scale='mel'
)

# Visualize frequency response
isac_fb.plot_response()
```
Condition number: 1.01
<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/ISAC_response.png?raw=true" width="100%">

```python
# Forward transform
y = isac_fb(x)
x_reconstructed = isac_fb.decoder(y)

# Visualize time-frequency representation
isac_fb.ISACgram(x, log_scale=True)
```

<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/ISAC_coeff.png?raw=true" width="100%">

### HybrA with Learnable Filters

```python
from hybra import HybrA

# Create hybrid filterbank with learnable components
hybra_fb = HybrA(
    kernel_size=1024,
    learned_kernel_size=23, 
    num_channels=128, 
    L=L, 
    fs=fs, 
    tighten=True
)

# Visualize frequency response
hybra_fb.plot_response()

# Check condition number for stability
print(f"Condition number: {hybra_fb.condition_number():.2f}")
```
Condition number: 1.06
<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/HybrA_response.png?raw=true" width="100%">

```python
# Forward pass (supports gradients for training)
y = hybra_fb(x)
x_reconstructed = hybra_fb.decoder(y)

# Visualize time-frequency representation
hybra_fb.ISACgram(x, log_scale=True)
```

<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/HybrA_coeff.png?raw=true" width="100%">

### ISAC Spectrograms and MFCCs

```python
from hybra import ISACSpec, ISACCC

# Spectrogram with temporal averaging for robust feature extraction
spectrogram = ISACSpec(
    kernel_size=1024,
    num_channels=40, 
    L=L, 
    fs=fs, 
    power=2.0,
    is_log=True
)

# MFCC-like cepstral coefficients for speech recognition
mfcc_extractor = ISACCC(
    kernel_size=1024,
    num_channels=40,
    num_cc=13, 
    L=L, 
    fs=fs
)

# Extract features
spec_coeffs = spectrogram(x)
mfcc_coeffs = mfcc_extractor(x)

print(f"Spectrogram shape: {spec_coeffs.shape}")
print(f"MFCC shape: {mfcc_coeffs.shape}")
```

### Integration with Neural Networks

Filterbanks can be easily integrated into neural networks as encoder/decoder pairs:
```python
import torch
import torch.nn as nn
import torchaudio
from hybra import HybrA

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_before = nn.Linear(40, 400)

        self.gru = nn.GRU(
            input_size=400,
            hidden_size=400,
            num_layers=2,
            batch_first=True,
        )

        self.linear_after = nn.Linear(400, 600)
        self.linear_after2 = nn.Linear(600, 600)
        self.linear_after3 = nn.Linear(600, 40)


    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = torch.relu(self.linear_before(x))
        x, _ = self.gru(x)
        x = torch.relu(self.linear_after(x))
        x = torch.relu(self.linear_after2(x))
        x = torch.sigmoid(self.linear_after3(x))
        x = x.permute(0, 2, 1)

        return x

class HybridfilterbankModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.nsnet = Net()
        self.fb = HybrA()

    def forward(self, x):
        x = self.fb(x)
        mask = self.nsnet(torch.log10(torch.max(x.abs()**2, 1e-8 * torch.ones_like(x, dtype=torch.float32))))
        return self.fb.decoder(x*mask)

if __name__ == '__main__':
    audio, fs = torchaudio.load('your_audio.wav') 
    model = HybridfilterbankModel()
    model(audio)
```

## Citation

If you find our work valuable and use HybrA or ISAC in your work, please cite

```
@inproceedings{haider2024holdmetight,
  author = {Haider, Daniel and Perfler, Felix and Lostanlen, Vincent and Ehler, Martin and Balazs, Peter},
  booktitle = {Annual Conference of the International Speech Communication Association (Interspeech)},
  year = {2024},
  title = {Hold me tight: Stable encoder/decoder design for speech enhancement},
}
@inproceedings{haider2025isac,
  author = {Haider, Daniel and Perfler, Felix and Balazs, Peter and Hollomey, Clara and Holighaus, Nicki},
  title = {{ISAC}: An Invertible and Stable Auditory Filter
  Bank with Customizable Kernels for ML Integration},
  booktitle = {International Conference on Sampling Theory and Applications (SampTA)},
  year = {2025}
}
```
