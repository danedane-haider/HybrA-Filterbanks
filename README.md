![Logo](https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/HybrA.png)

## About
This repository contains the official implementaions of [Hybrid Auditory filterbanks](https://arxiv.org/abs/2408.17358) and [ISAC](https://arxiv.org/abs/2505.07709). The modules are designed to be easily usable in the design of PyTorch model designs.

## Documentation
[https://github.com/danedane-haider/HybrA-Filterbanks](https://danedane-haider.github.io/HybrA-Filterbanks/main/)

## Installation
We publish all releases on PyPi. You can install the current version by running:
```
pip install hybra
```

## Usage
This package offers several PyTorch modules to be used in your code performing transformations of an input signal into a time frequency representation.
```python
import torchaudio
from hybra import HybrA, ISAC

x, fs = torchaudio.load("audio.wav")

isac_filterbank = ISAC(fs=fs)
y = isac_filterbank(x)
isac_filterbank.plot_response()

hybra_filterbank = HybrA(fs=fs)
y = hybra_filterbank(x)
hybra_filterbank.plot_response()
```

It is also straightforward to include the filterbank in a model, e.g. as a encoder/decoder pair.
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

If you find our work valuable, please cite

```
@article{HaiderTight2024,
  title={Hold me Tight: Trainable and stable hybrid auditory filterbanks for speech enhancement},
  author={Haider, Daniel and Perfler, Felix and Lostanlen, Vincent and Ehler, Martin and Balazs, Peter},
  journal={arXiv preprint arXiv:2408.17358},
  year={2024}
}
@article{HaiderISAC2025,
      title={ISAC: An Invertible and Stable Auditory Filter Bank with Customizable Kernels for ML Integration}, 
      author={Daniel Haider and Felix Perfler and Peter Balazs and Clara Hollomey and Nicki Holighaus},
      year={2025},
      url={arXiv preprint arXiv:2505.07709}, 

}
```
