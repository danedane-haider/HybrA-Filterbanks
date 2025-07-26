.. HybrA-Filterbanks documentation master file, created by
   sphinx-quickstart on Wed May 21 11:34:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HybrA-Filterbanks
=================

About
-----
This repository contains the official implementations of `HybrA <https://arxiv.org/abs/2408.17358>`_ and `ISAC <https://arxiv.org/abs/2505.07709>`_. ISAC is an invertible and stable auditory filterbank with customizable kernel size, and HybrA extends ISAC via an additional set of learnable kernels. The two filterbanks are implemented as PyTorch nn.Module and therefore easily integrable into any neural network. As an essential mathematical foundation for the construction of ISAC and HybrA, the repository contains many fast frame-theoretic functions, such as the computation of framebounds, aliasing terms, and regularizers for tightening. 

Installation
------------

We publish all releases on PyPi. You can install the current version by
running:

::

   pip install hybra

Usage
-----

Construct an ISAC and HybrA filterbank, and plot the filter frequency responses. Transform an input audio signal into the corresponding learnable time-frequency representation, and plot it.

.. code-block:: python
   :linenos:
   :caption: ISAC / HybrA example

   import torchaudio
   from hybra import ISAC, HybrA, ISACgram

   x, fs = torchaudio.load("your_audio.wav")
   x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
   L = x.shape[-1]

   isac_fb = ISAC(kernel_size=1024, num_channels=128, L=L, fs=fs)
   isac_fb.plot_response()

   y = isac_fb(x)
   x_tilde = isac_fb.decoder(y)
   ISACgram(y, isac_fb.fc, L=L, fs=fs)

It is also straightforward to include them in any model, e.g., as an encoder/decoder pair.

.. code-block:: python
   :linenos:
   :caption: HybrA model example

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

Citation
--------

If you find our work valuable, please cite

::

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



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

