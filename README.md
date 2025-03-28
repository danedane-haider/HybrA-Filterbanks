![Logo](https://github.com/danedane-haider/HybrA-Filterbanks/blob/main/HybrA.png)

## About
This repository contains the official implementaions of [Hybrid Auditory filterbanks](https://arxiv.org/abs/2408.17358) and [ISAC](https://arxiv.org/abs/). The modules are designed to be easily usable in the design of PyTorch model designs.

## Installation
We publish all releases on PyPi. You can install the current version by running:
```
pip install hybra
```

## Usage
This package offers several PyTorch modules to be used in your code performing transformations of an input signal into a time frequency representation.
```python
from hybra import HybrA

import soundfile
import torch

device = "mps"

x, fs = soundfile.read("./audio/crush.wav")
x = 2 * torch.tensor(x[:, 0], dtype=torch.float32).unsqueeze(0)
sig_len = x.shape[-1]

filterbank = HybrA(L=sig_len,stride=8,scale='mel').to(device)
filterbank.plot_response()

out = filterbank(x.to(device))
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
```
