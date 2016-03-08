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
   
.. doxygenclass:: ReLe::Core
   :members:
   
.. doxygenfunction:: ReLe::buildCore      
   
.. doxygenclass:: ReLe::BatchOnlyCore
   :members:   
   
.. doxygenfunction:: ReLe::buildBatchOnlyCore      
   
.. doxygenclass:: ReLe::BatchCore
   :members:      
   
.. doxygenfunction:: ReLe::buildBatchCore      
   
.. doxygenclass:: ReLe::Solver
   :members:   
   
Basic Environments
------------------

.. doxygenclass:: ReLe::FiniteMDP
   :members:   

.. doxygenclass:: ReLe::DenseMDP
   :members:   
   
.. doxygenclass:: ReLe::ContinuousMDP
   :members:      
   
Core loggers
------------
.. doxygenclass:: ReLe::Logger
   :members:
   
.. doxygenclass:: ReLe::LoggerStrategy
   :members:
   :protected-members:
   
.. doxygenclass:: ReLe::PrintStrategy
   :members:
   
.. doxygenclass:: ReLe::WriteStrategy
   :members:

.. doxygenclass:: ReLe::EvaluateStrategy
   :members:   
   
.. doxygenclass:: ReLe::CollectorStrategy
   :members:   
   
Batch loggers
-------------

.. doxygenclass:: ReLe::BatchAgentLogger
   :members:
   :protected-members:

.. doxygenclass:: ReLe::BatchAgentPrintLogger
   :members:

.. doxygenclass:: ReLe::BatchDatasetLogger
   :members:
   
.. doxygenclass:: ReLe::CollectBatchDatasetLogger
   :members:
   
.. doxygenclass:: ReLe::WriteBatchDatasetLogger
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
---------------

.. doxygenclass:: ReLe::GenericMVNPolicy

.. doxygenclass:: ReLe::GenericMVNDiagonalPolicy

.. doxygenclass:: ReLe::GenericMVNStateDependantStddevPolicy
