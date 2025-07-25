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

x, fs = torchaudio.load("/Users/felixperfler/Downloads/sweep.wav")
x = x.unsqueeze(0)

isac_filterbank = ISAC(fs=fs)
y = isac_filterbank(x)
isac_filterbank.plot_response()

hybra_filterbank = HybrA(fs=fs)
y = hybra_filterbank(x)
hybra_filterbank.plot_response()
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
