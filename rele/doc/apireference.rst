.. _apireference:

.. warning::

    Please be advised that the reference documentation discussing ReLe
    internals is currently incomplete. Please refer to the previous sections
    and the ReLe header files for the nitty gritty details.

=============
API Reference
=============

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
   
.. doxygenclass:: ReLe::Ensemble_
   :members:
   
.. doxygenclass:: ReLe::ParametricRegressor_
   :members:
   
.. doxygenclass:: ReLe::BatchRegressor_
   :members:
   
.. doxygenclass:: ReLe::UnsupervisedBatchRegressor_
   :members:
   
Basis functions
---------------

.. doxygenclass:: ReLe::AffineFunction
   :members:

.. doxygenclass:: ReLe::GaussianRbf
   :members:
   
.. doxygenclass:: ReLe::IdentityBasis_
   :members:
   
.. doxygenclass:: ReLe::IdentityBasis
   :members:
   
.. doxygenclass:: ReLe::FiniteIdentityBasis
   :members:
   
.. doxygenclass:: ReLe::VectorFiniteIdentityBasis
   :members:

.. doxygenclass:: ReLe::InverseBasis_
   :members:

Features types
--------------

.. doxygenclass:: ReLe::DenseFeatures_
   :members:
   
.. doxygenclass:: ReLe::SparseFeatures_
   :members:
   
.. doxygenclass:: ReLe::TilesCoder_
   :members:
   
Batch Dataset Utils
-------------------
.. doxygenclass:: ReLe::BatchDataRaw_
   :members:
.. doxygenclass:: ReLe::BatchData_
   :members:
.. doxygenclass:: ReLe::MiniBatchData_
   :members:
.. doxygenclass:: ReLe::BatchDataSimple_
   :members:

.. doxygenclass:: ReLe::Normalization
   :members:
.. doxygenclass:: ReLe::NoNormalization
   :members:
.. doxygenclass:: ReLe::MinMaxNormalization
   :members:
.. doxygenclass:: ReLe::ZscoreNormalization
   :members:


.. doxygenfunction:: ReLe::normalizeDataset
.. doxygenfunction:: ReLe::normalizeDatasetFull




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

Distributions
=============

.. doxygenclass:: ReLe::Distribution
   :members:
   :protected-members:

.. doxygenclass:: ReLe::DifferentiableDistribution
   :members:
   
.. doxygenclass::  ReLe::FisherInterface
   :members:
   
Normal Distributions
--------------------

.. doxygenclass:: ReLe::ParametricNormal
   :members:
   
.. doxygenclass:: ReLe::ParametricDiagonalNormal
   :members:   
   
.. doxygenclass:: ReLe::ParametricLogisticNormal
   :members:
   
.. doxygenclass:: ReLe::ParametricCholeskyNormal
   :members:
   
.. doxygenclass:: ReLe::ParametricFullNormal
   :members:
   
Wishart Distributions
---------------------

.. doxygenclass:: ReLe::WishartBase
   :members:
   
.. doxygenclass:: ReLe::Wishart
   :members:
   
.. doxygenclass:: ReLe::InverseWishart
   :members:   


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
   
Algorithms
==========

Step Rules
----------

Learning Rate
^^^^^^^^^^^^^

.. doxygenclass:: ReLe::LearningRate_
   :members:
   
.. doxygenclass:: ReLe::ConstantLearningRate_
   :members:
   
.. doxygenclass:: ReLe::DecayingLearningRate_
   :members:      
   
   
Gradient Step
^^^^^^^^^^^^^

.. doxygenclass:: ReLe::GradientStep
   :members:
   :protected-members:
   
.. doxygenclass:: ReLe::ConstantGradientStep
   :members:
   
.. doxygenclass:: ReLe::VectorialGradientStep
   :members:
   
.. doxygenclass:: ReLe::AdaptiveGradientStep
   :members:
   
Batch
-----

.. doxygenclass:: ReLe::FQI
   :members:
   
.. doxygenclass:: ReLe::DoubleFQI
   :members:
   
.. doxygenclass:: ReLe::DoubleFQIEnsemble
   :members:

.. doxygenclass:: ReLe::W_FQI
   :members:
   
.. doxygenclass:: ReLe::LSPI
   :members:

Temporal Difference
-------------------

.. doxygenclass:: ReLe::FiniteTD
   :members:
   :protected-members:

.. doxygenclass:: ReLe::LinearTD
   :members:
   :protected-members:
   
.. doxygenclass:: ReLe::SARSA
   :members:
   
.. doxygenclass:: ReLe::SARSA_lambda
   :members:
   
.. doxygenclass:: ReLe::Q_Learning
   :members:
   
.. doxygenclass:: ReLe::DoubleQ_Learning
   :members:
   
.. doxygenclass:: ReLe::WQ_Learning
   :members:
   
.. doxygenclass:: ReLe::R_Learning
   :members:
   
.. doxygenclass:: ReLe::LinearGradientSARSA
   :members:

Output Data
^^^^^^^^^^^

.. doxygenclass:: ReLe::FiniteTDOutput
   :members:

.. doxygenclass:: ReLe::LinearTDOutput
   :members:   

.. doxygenclass:: ReLe::R_LearningOutput
   :members:

.. doxygenclass:: ReLe::FQIOutput
   :members:   

Batch
-----

Policy Search
-------------   

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
   
Utils
=====

Armadillo Extensions
--------------------

Linear Algebra
^^^^^^^^^^^^^^

.. doxygenfunction:: ReLe::null

.. doxygenfunction:: ReLe::rref

.. doxygenfunction:: ReLe::wrapTo2Pi(double)

.. doxygenfunction:: ReLe::wrapTo2Pi(const arma::vec&)

.. doxygenfunction:: ReLe::wrapToPi(double)

.. doxygenfunction:: ReLe::wrapToPi(const arma::vec&)

.. doxygenfunction:: ReLe::meshgrid

.. doxygenfunction:: ReLe::blockdiagonal(const std::vector<arma::mat>&)

.. doxygenfunction:: ReLe::blockdiagonal(const std::vector<arma::mat>&, int, int)

.. doxygenfunction:: ReLe::range

.. doxygenfunction:: ReLe::vecToTriangular

.. doxygenfunction:: ReLe::triangularToVec

.. doxygenfunction:: ReLe::safeChol

Distributions
^^^^^^^^^^^^^

.. doxygenfunction:: ReLe::mvnpdf(const arma::vec&, const arma::vec&, const arma::mat&)

.. doxygenfunction:: ReLe::mvnpdfFast(const arma::vec&, const arma::vec&, const arma::mat&, const double&)

.. doxygenfunction:: ReLe::mvnpdf(const arma::vec&, const arma::vec&, const arma::mat&, arma::vec&)

.. doxygenfunction:: ReLe::mvnpdf(const arma::vec&, const arma::vec&, const arma::mat&, arma::vec&, arma::mat&)

.. doxygenfunction:: ReLe::mvnpdfFast(const arma::vec&, const arma::vec&, const arma::mat&, const double&, arma::vec&, arma::vec&)

.. doxygenfunction:: ReLe::mvnpdf(const arma::mat&, const arma::vec&, const arma::mat&, arma::vec&)

.. doxygenfunction:: ReLe::mvnrand(int, const arma::vec&, const arma::mat&)

.. doxygenfunction:: ReLe::mvnrand(const arma::vec&, const arma::mat&)

.. doxygenfunction:: ReLe::mvnrandFast(const arma::vec&, const arma::mat&)


Other Utils
-----------

.. doxygenclass:: ReLe::ConsoleManager
   :members:
   
.. doxygenclass:: ReLe::CSVutils
   :members:
   
.. doxygenclass:: ReLe::FileManager
   :members:
   
.. doxygenclass:: ReLe::RngGenerators
   :members:
   
.. doxygenclass:: ReLe::RandomGenerator
   :members:
   
.. doxygenclass:: ReLe::Range
   :members:
   
.. doxygenclass:: ReLe::ModularRange
   :members:
   
.. doxygenclass:: ReLe::NumericalGradient
   :members:
   
.. doxygenclass:: ReLe::Range2Pi
   :members:

.. doxygenclass:: ReLe::RangePi
   :members: