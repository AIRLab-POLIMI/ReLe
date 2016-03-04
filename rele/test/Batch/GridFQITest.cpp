/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Written by: Carlo D'Eramo
 */

#include "rele/core/Core.h"
#include "rele/core/BatchCore.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/algorithms/batch/td/DoubleFQI.h"
#include "rele/algorithms/td/Q-Learning.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/core/FiniteMDP.h"
#include "rele/generators/GridWorldGenerator.h"
#include "rele/approximators/regressors/nn/FFNeuralNetwork.h"
#include "rele/approximators/regressors/nn/FFNeuralNetworkEnsemble.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/regressors/trees/KDTree.h"
#include "rele/approximators/regressors/trees/ExtraTree.h"
#include "rele/approximators/regressors/trees/ExtraTreeEnsemble.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    GridWorldGenerator generator;
    generator.load(argv[1]);

    double gamma = 0.9;
    FiniteMDP&& mdp = generator.getMDP(gamma);

    unsigned int nActions = mdp.getSettings().finiteActionDim;
    unsigned int nStates = mdp.getSettings().finiteStateDim;

    BasisFunctions bfs;
    bfs = IdentityBasis::generate(2);

    DenseFeatures phi(bfs);

    //FFNeuralNetwork QRegressorA(phi, 50, 1);
    //FFNeuralNetwork QRegressorB(phi, 50, 1);
    //QRegressorA.getHyperParameters().lambda = 0.0005;
    //QRegressorA.getHyperParameters().maxIterations = 10;
    //QRegressorB.getHyperParameters().maxIterations = 10;
    //QRegressorB.getHyperParameters().maxIterations = 10;

    arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> QRegressorA(phi, defaultNode, 1, 1);
    KDTree<arma::vec, arma::vec> QRegressorB(phi, defaultNode, 1, 1);

    FQI<FiniteState> batchAgent(QRegressorA, nStates, nActions, 1e-8);
    //DoubleFQI<FiniteState> batchAgent(QRegressorA, QRegressorB, nStates, nActions, 0.9, 1e-8);

    auto&& core = buildBatchCore(mdp, batchAgent);

    core.getSettings().envName = "gw";
    core.getSettings().algName = "fqi";
    core.getSettings().nEpisodes = 100;
    core.getSettings().nTransitions = 1000;
    core.getSettings().episodeLength = 100;
    core.getSettings().maxBatchIterations = 100;

    arma::mat Q(nStates, nActions, arma::fill::zeros);
    e_Greedy policy;
    policy.setQ(&Q);

    core.run(policy);

    return 0;
}
