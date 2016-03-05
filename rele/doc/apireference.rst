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
   

Basic Interfaces
----------------

.. doxygenclass:: ReLe::Environment
   :members:
   :protected-members: 

.. doxygenclass:: ReLe::Agent
   :members:
   :protected-members:

.. doxygenclass:: ReLe::BatchAgent
   :members:
   :protected-members:
   
.. doxygenclass:: ReLe::Solver
   :members:   
   
Basic Environments
------------

.. doxygenclass:: ReLe::FiniteMDP
   :members:   

.. doxygenclass:: ReLe::DenseMDP
   :members:   
   
.. doxygenclass:: ReLe::ContinuosMDP
   :members:      

Basic Utilities
---------------

.. doxygenclass:: ReLe::PolicyEvalAgent
   :members:
   
.. doxygenclass:: ReLe::PolicyEvalDistribution
   :members:   

Policy Representations
======================

.. doxygenclass:: ReLe::Policy
   :members:

.. doxygenclass:: ReLe::ParametricPolicy
   :members:

.. doxygenclass:: ReLe::DifferentiablePolicy
   :members:

Normal Policies
-----------------------

.. doxygenclass:: ReLe::GenericMVNPolicy

.. doxygenclass:: ReLe::GenericMVNDiagonalPolicy

.. doxygenclass:: ReLe::GenericMVNStateDependantStddevPolicy
