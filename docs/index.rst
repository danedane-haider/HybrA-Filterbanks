.. HybrA-Filterbanks documentation master file, created by
   sphinx-quickstart on Wed May 21 11:34:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HybrA-Filterbanks
=================

**Auditory-inspired filterbanks for deep learning**

Welcome to HybrA-Filterbanks, a PyTorch library providing state-of-the-art auditory-inspired filterbanks for audio processing and deep learning applications.

Overview
--------

This library contains the official implementations of:

* **ISAC** (`paper <https://arxiv.org/abs/2505.07709>`_): Invertible and Stable Auditory filterbank with Customizable kernels for ML integration
* **HybrA** (`paper <https://arxiv.org/abs/2408.17358>`_): Hybrid Auditory filterbank that extends ISAC with learnable filters
* **ISACSpec**: Spectrogram variant with temporal averaging for robust feature extraction  
* **ISACCC**: Cepstral coefficient extractor for speech recognition applications

.. figure:: _static/hybra-magnitude-training.gif
   :alt: HybrA magnitude training animation
   :align: center
   :width: 720px

   Magnitude learning dynamics of HybrA during training.

Key Features
------------

✨ **PyTorch Integration**: All filterbanks are implemented as ``nn.Module`` for seamless integration into neural networks

🎯 **Auditory Modeling**: Based on human auditory perception principles (mel, ERB, bark scales)

⚡ **Fast Implementation**: Optimized using FFT-based circular convolution

🔧 **Flexible Configuration**: Customizable kernel sizes, frequency ranges, and scales

📊 **Frame Theory**: Built-in functions for frame bounds, condition numbers, and stability analysis

🎨 **Visualization**: Rich plotting capabilities for filter responses and time-frequency representations 

Installation
------------

We publish all releases on PyPi. You can install the current version by
running:

::

   pip install hybra

Quick Start
-----------

Basic ISAC Filterbank
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import torch
   from hybra import ISAC

   # Create ISAC filterbank
   filterbank = ISAC(
       kernel_size=128, 
       num_channels=40, 
       fs=16000, 
       L=16000, 
       scale='mel'
   )

   # Process audio signal
   x = torch.randn(1, 16000)  # Random signal for demo
   coefficients = filterbank(x)
   reconstructed = filterbank.decoder(coefficients)

   # Visualize
   filterbank.plot_response()
   filterbank.ISACgram(x, log_scale=True)

HybrA with Learnable Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   from hybra import HybrA

   # Create hybrid filterbank with learnable components
   hybrid_fb = HybrA(
       kernel_size=128,
       learned_kernel_size=23,
       num_channels=40,
       fs=16000,
       L=16000
   )

   # Forward pass (supports gradients)
   x = torch.randn(1, 16000, requires_grad=True)
   y = hybrid_fb(x)

   # Check condition number for stability
   print(f"Condition number: {hybrid_fb.condition_number():.2f}")

ISAC Spectrograms and MFCCs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   from hybra import ISACSpec, ISACCC

   # Spectrogram with temporal averaging
   spectrogram = ISACSpec(
       num_channels=40, 
       fs=16000, 
       L=16000, 
       power=2.0,
       is_log=True
   )

   # MFCC-like cepstral coefficients  
   mfcc_extractor = ISACCC(
       num_channels=40,
       num_cc=13,
       fs=16000,
       L=16000
   )

   x = torch.randn(1, 16000)
   spec = spectrogram(x)
   mfccs = mfcc_extractor(x)

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
   :caption: Documentation:

   api
   examples
   mathematical_background

.. toctree::
   :maxdepth: 1
   :caption: Links:

   GitHub Repository <https://github.com/danedane-haider/HybrA-Filterbanks>
   PyPI Package <https://pypi.org/project/hybra/>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
