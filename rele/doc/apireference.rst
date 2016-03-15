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
   :protected-members:
   
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
   
Reward Transformation
---------------------

.. doxygenclass:: ReLe::RewardTransformation
   :members:
   
.. doxygenclass:: ReLe::IndexRT
   :members:
   
.. doxygenclass:: ReLe::WeightedSumRT
   :members:  
   

Basic Utilities
---------------

.. doxygenclass:: ReLe::PolicyEvalAgent
   :members:
   
.. doxygenclass:: ReLe::PolicyEvalDistribution
   :members:   
   
Function Approximators
======================

Basic interfaces
----------------

.. doxygenclass:: ReLe::BasisFunction_
   :members:

.. doxygenclass:: ReLe::Tiles_
   :members:
    
.. doxygenclass:: ReLe::Features_
   :members:

.. doxygenclass:: ReLe::Regressor_
   :members:
   
.. doxygenclass:: ReLe::ParametricRegressor_
   :members:
   
.. doxygenclass:: ReLe::BatchRegressor_
   :members:
   
.. doxygenclass:: ReLe::UnsupervisedBatchRegressor_
   :members:

Features types
--------------

.. doxygenclass:: ReLe::DenseFeatures_
   :members:
   
.. doxygenclass:: ReLe::SparseFeatures_
   :members:
   
.. doxygenclass:: ReLe::TilesCoder_
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

Environments
============

.. doxygenclass:: ReLe::Dam
   :members:
   
.. doxygenclass:: ReLe::DeepSeaTreasure
   :members:

.. doxygenclass:: ReLe::GaussianRewardMDP
   :members:
   
.. doxygenclass:: ReLe::LQR
   :members:
   
.. doxygenclass:: ReLe::MountainCar
   :members:
   
.. doxygenclass:: ReLe::MultiHeat
   :members:
   
.. doxygenclass:: ReLe::NLS
   :members:
   
.. doxygenclass:: ReLe::Portfolio
   :members:

.. doxygenclass:: ReLe::Pursuer
   :members:
   
.. doxygenclass:: ReLe::Rocky
   :members:
   
.. doxygenclass:: ReLe::Segway
   :members:
   
.. doxygenclass:: ReLe::ShipSteering
   :members:
   
.. doxygenclass:: ReLe::DiscreteActionSwingUp
   :members:

.. doxygenclass:: ReLe::TaxiFuel
   :members:
   
.. doxygenclass:: ReLe::UnderwaterVehicle
   :members:
   
.. doxygenclass:: ReLe::UnicyclePolar
   :members:
   
Generators
----------

.. doxygenclass:: ReLe::FiniteGenerator
   :members:
   :protected-members:
   
.. doxygenclass:: ReLe::SimpleChainGenerator
   :members:

.. doxygenclass:: ReLe::GridWorldGenerator
   :members:

Optimization
============
.. doxygenclass:: ReLe::Optimization
   :members:

.. doxygenclass:: ReLe::Simplex
   :members:
   
Feature Selection
=================

.. doxygenclass:: ReLe::FeatureSelectionAlgorithm
   :members:
   
.. doxygenclass:: ReLe::LinearFeatureSelectionAlgorithm
   :members:

.. doxygenclass:: ReLe::PrincipalComponentAnalysis
   :members:
   
.. doxygenclass:: ReLe::PrincipalFeatureAnalysis
   :members:
