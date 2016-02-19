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
#include "rele/utils/FileManager.h"
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


// This simple test is used to verify the correctness of the FQI implementation
int main(int argc, char *argv[])
{
    bool acquireData = true;

    FileManager fm("gw", "FQI");
    fm.createDir();
    fm.cleanDir();

    // A grid world is generated
    GridWorldGenerator generator;
    generator.load(argv[1]);

    // The MDP w.r.t. the generated grid world is extracted
    FiniteMDP&& mdp = generator.getMDP(1.0);

    unsigned int nActions = mdp.getSettings().finiteActionDim;
    unsigned int nStates = mdp.getSettings().finiteStateDim;

    /* Decide whether to acquire data with a policy, or to use already
     * collected ones.
     */
    Dataset<FiniteAction, FiniteState> data;
    if(acquireData)
    {
        // Policy declaration
        e_Greedy policy;
        policy.setEpsilon(0.25);
        policy.setNactions(nActions);

        // The agent is instantiated. It takes the policy as parameter.
        Q_Learning expert(policy);

        /* The Core class is what ReLe uses to move the agent in the MDP. It is
         *  instantiated using the MDP and the agent itself.
         */
        auto&& expertCore = buildCore(mdp, expert);

        /* The CollectorStrategy is used to collect data from the agent that is
         * moving in the MDP. Here, it is used to store the transition that are used
         * as the inputs of the dataset provided to FQI.
         */
        CollectorStrategy<FiniteAction, FiniteState> collection;
        expertCore.getSettings().loggerStrategy = &collection;

        // Number of transitions in an episode
        unsigned int nTransitions = 1000;
        expertCore.getSettings().episodeLength = nTransitions;
        // Number of episodes
        unsigned int nEpisodes = 100;
        expertCore.getSettings().episodeN = nEpisodes;

        /* The agent start the exploration that will last for the provided number of
         *  episodes.
         */
        expertCore.runEpisodes();

        // The dataset is build from the data collected by the CollectorStrategy.
        data = collection.data;

        // Dataset is written into a file
        ofstream out(fm.addPath("dataset.csv"), ios_base::out);
        out << std::setprecision(OS_PRECISION);
        if(out.is_open())
            data.writeToStream(out);
        out.close();

        cout << endl << "# Ended data collection and save" << endl << endl;
    }
    else
    {
        // Dataset is loaded from a file
        ifstream in(fm.addPath("dataset.csv"), ios_base::in);
        in >> std::setprecision(OS_PRECISION);
        if(in.is_open())
            data.readFromStream(in);
        in.close();
    }

    /* The basis functions for the features of the regressor are created here.
     * IdentityBasis functions are features that simply replicate the input. We can use
     * Manhattan distance from s' to goal as feature of the input vector (state, action).
     */
    BasisFunctions bfs;
    bfs = IdentityBasis::generate(2);

    // The feature vector is build using the chosen basis functions
    DenseFeatures phi(bfs);

    // ******* FQI *******

    // Neural Network
    //FFNeuralNetwork QRegressorA(phi, 50, 1);
    //FFNeuralNetwork QRegressorB(phi, 50, 1);
    //QRegressorA.getHyperParameters().lambda = 0.0005;
    //QRegressorA.getHyperParameters().maxIterations = 10;
    //QRegressorB.getHyperParameters().maxIterations = 10;
    //QRegressorB.getHyperParameters().maxIterations = 10;

    // Tree
    arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> QRegressorA(phi, defaultNode, 1, 1);
    KDTree<arma::vec, arma::vec> QRegressorB(phi, defaultNode, 1, 1);

    //FQI<FiniteState> fqi(QRegressorA, nStates, nActions, 0.9);
    DoubleFQI<FiniteState> fqi(QRegressorA, QRegressorB, nStates, nActions, 0.9);

    auto&& core = buildCore(data, fqi);

    cout << "Starting FQI..." << endl;

    for(unsigned int i = 0; i < 4; i++)
    {
    	core.getSettings().maxIterations = 1000;
    	core.getSettings().epsilon = 1e-8;

    	core.runEpisode();
    }

    return 0;
}
