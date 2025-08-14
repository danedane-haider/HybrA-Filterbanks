API Reference
=============

This page contains the complete API reference for all classes and functions in the HybrA-Filterbanks library.

Core Filterbanks
----------------

ISAC - Invertible and Stable Auditory Filterbank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hybra.ISAC
   :members:
   :show-inheritance:
   :exclude-members: training, eval, load_state_dict, state_dict, parameters, named_parameters, children, named_children, modules, named_modules, apply, cuda, cpu, type, float, double, half, to, register_backward_hook, register_forward_pre_hook, register_forward_hook, add_module, get_submodule, get_parameter, get_buffer, _get_name, extra_repr, _load_from_state_dict, _save_to_state_dict, dump_patches

HybrA - Hybrid Auditory Filterbank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hybra.HybrA
   :members:
   :show-inheritance:
   :exclude-members: training, eval, load_state_dict, state_dict, parameters, named_parameters, children, named_children, modules, named_modules, apply, cuda, cpu, type, float, double, half, to, register_backward_hook, register_forward_pre_hook, register_forward_hook, add_module, get_submodule, get_parameter, get_buffer, _get_name, extra_repr, _load_from_state_dict, _save_to_state_dict, dump_patches

Spectrogram and Cepstral Variants
---------------------------------

ISACSpec - ISAC Spectrogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hybra.ISACSpec
   :members:
   :show-inheritance:
   :exclude-members: training, eval, load_state_dict, state_dict, parameters, named_parameters, children, named_children, modules, named_modules, apply, cuda, cpu, type, float, double, half, to, register_backward_hook, register_forward_pre_hook, register_forward_hook, add_module, get_submodule, get_parameter, get_buffer, _get_name, extra_repr, _load_from_state_dict, _save_to_state_dict, dump_patches

ISACCC - ISAC Cepstral Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hybra.ISACCC
   :members:
   :show-inheritance:
   :exclude-members: training, eval, load_state_dict, state_dict, parameters, named_parameters, children, named_children, modules, named_modules, apply, cuda, cpu, type, float, double, half, to, register_backward_hook, register_forward_pre_hook, register_forward_hook, add_module, get_submodule, get_parameter, get_buffer, _get_name, extra_repr, _load_from_state_dict, _save_to_state_dict, dump_patches

Utility Functions
-----------------

Frame Theory Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hybra.utils.frame_bounds
.. autofunction:: hybra.utils.condition_number

Auditory Scale Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hybra.utils.freqtoaud
.. autofunction:: hybra.utils.audtofreq
.. autofunction:: hybra.utils.audspace
.. autofunction:: hybra.utils.audspace_mod

Filterbank Construction
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hybra.utils.audfilters
.. autofunction:: hybra.utils.fctobw
.. autofunction:: hybra.utils.bwtofc
.. autofunction:: hybra.utils.firwin
.. autofunction:: hybra.utils.modulate

Convolution Operations
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hybra.utils.circ_conv
.. autofunction:: hybra.utils.circ_conv_transpose

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hybra.utils.ISACgram
.. autofunction:: hybra.utils.plot_response
.. autofunction:: hybra.utils.response

Frame Analysis Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hybra.utils.frequency_correlation
.. autofunction:: hybra.utils.alias
.. autofunction:: hybra.utils.can_tight
.. autofunction:: hybra.utils.fir_tightener3000

Helper Functions
~~~~~~~~~~~~~~~~

.. autofunction:: hybra.utils.upsample
.. autofunction:: hybra.utils.freqtoaud_mod
.. autofunction:: hybra.utils.audtofreq_mod

