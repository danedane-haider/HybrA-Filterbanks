![Logo](https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/HybrA.png)

## About
This repository contains the official implementations of [HybrA](https://arxiv.org/abs/2408.17358) and [ISAC](https://arxiv.org/abs/2505.07709). ISAC is an invertible and stable auditory filterbank with customizable kernel size, and HybrA extends ISAC via an additional set of learnable kernels. The two filterbanks are implemented as PyTorch nn.Module and therefore easily integrable into any neural network. As an essential mathematical foundation for the construction of ISAC and HybrA, the repository contains many fast frame-theoretic functions, such as the computation of framebounds, aliasing terms, and regularizers for tightening. 

## Documentation
[https://github.com/danedane-haider/HybrA-Filterbanks](https://danedane-haider.github.io/HybrA-Filterbanks/main/)

## Installation
We publish all releases on PyPi. You can install the current version by running:
```
pip install hybra
```

## Usage
Construct an ISAC and HybrA filterbank, and plot the filter frequency responses. Transform an input audio signal into the corresponding learnable time-frequency representation, and plot it.
```python
import torchaudio
from hybra import ISAC, HybrA, ISACgram

x, fs = torchaudio.load("audio.wav")
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
L = x.shape[-1]

isac_filterbank = ISAC(kernel_size=1024, num_channels=128, L=L, fs=fs)
isac_filterbank.plot_response()
```

<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/ISAC_response.png?raw=true" width="100%">

```python
y = isac_filterbank(x)
x_tilde = isac_filterbank.decoder(y)
ISACgram(y)
```

<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/ISAC_coeff.png?raw=true" width="100%">

```python

hybra_filterbank = HybrA(kernel_size=1024, learned_kernel_size=23, num_channels=128, L=L, fs=fs, tighten=True)
hybra_filterbank.plot_response()
```

<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/HybrA_response.png?raw=true" width="100%">

```python
y = hybra_filterbank(x)
x_tilde = hybra_filterbank.decoder(y)
ISACgram(y)
```

<img src="https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/plots/HybrA_coeff.png?raw=true" width="100%">

```python
y = hybra_filterbank(x)
ISACgram(y)
```

It is also straightforward to include them in your model, e.g., as an encoder/decoder pair.
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
        self.filterbank = HybrA()

    def forward(self, x):
        x = self.filterbank(x)
        mask = self.nsnet(torch.log10(torch.max(x.abs()**2, 1e-8 * torch.ones_like(x, dtype=torch.float32))))
        return self.filterbank.decoder(x*mask)

if __name__ == '__main__':
    audio, fs = torchaudio.load('audio.wav') 
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
