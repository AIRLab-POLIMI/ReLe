.. _apireference:

.. warning::

    Please be advised that the reference documentation discussing ReLe
    internals is currently incomplete. Please refer to the previous sections
    and the ReLe header files for the nitty gritty details.


API Reference
######################

.. highlight:: c++

All functions and classes provided by the C++ Format library reside
in namespace ``ReLe``.

Library Core
============

Basic concepts
--------------

.. doxygenclass:: ReLe::Action
   :members:

.. doxygenclass:: ReLe::FiniteAction
   :members:
   
.. doxygenclass:: ReLe::DenseAction
   :members:
   
.. doxygenclass:: ReLe::State
   :members:
   
.. doxygenclass:: ReLe::FiniteState
   :members:
   
.. doxygenclass:: ReLe::DenseState
   :members:

.. doxygenstruct:: ReLe::EnvironmentSettings
   :members:   
   
.. doxygenclass:: ReLe::AgentOutputData
   :members:    
   
.. doxygenstruct:: ReLe::action_type
   :members:
   
.. doxygenstruct:: ReLe::state_type
   :members:      

Trajectories
------------

.. doxygenstruct:: ReLe::Transition
   :members: 
   
.. doxygenclass:: ReLe::Episode
   :members:       

.. doxygenclass:: ReLe::Dataset
   :members:       


