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

#include "Core.h"
#include "PolicyEvalAgent.h"
#include "q_policy/e_Greedy.h"
#include "batch/FQI.h"
#include "features/DenseFeatures.h"
#include "FileManager.h"
#include "FiniteMDP.h"
#include "GridWorldGenerator.h"
#include "regressors/FFNeuralNetwork.h"
#include "basis/IdentityBasis.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;


// This simple test is used to verify the correctness of the FQI implementation
int main(int argc, char *argv[])
{
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

    /* This policy is used by an evaluation agent that has the purpose to
     *  build a dataset from its exploration of the environment. The policy is
     *  a random policy, thus allowing pure exploration. */
    e_Greedy randomPolicy;
    randomPolicy.setEpsilon(1.0);
    randomPolicy.setNactions(nActions);
    arma::mat Q(nStates, nActions);
    randomPolicy.setQ(&Q);

    // The agent is instantiated. It takes the policy as parameter.
    PolicyEvalAgent<FiniteAction, FiniteState> expert(randomPolicy);

    /* The Core class is what ReLe uses to move the agent in the MDP. It is
     *  instantiated using the MDP and the agent itself. */
    Core<FiniteAction, FiniteState> expertCore(mdp, expert);

    /* The CollectorStrategy is used to collect data from the agent that is
     *  moving in the MDP. Here, it is used to store the transition that are used
     *  as the inputs of the dataset provided to FQI. */
    CollectorStrategy<FiniteAction, FiniteState> collection;
    expertCore.getSettings().loggerStrategy = &collection;

    // Number of transitions in an episode
    unsigned int nTransitions = 250;
    expertCore.getSettings().episodeLength = nTransitions;
    // Number of episodes
    unsigned int nEpisodes = 100;
    expertCore.getSettings().testEpisodeN = nEpisodes;

    /* The agent start the exploration that will last for the provided number of
     *  episodes. */
    expertCore.runTestEpisodes();

    // The dataset is build from the data collected by the CollectorStrategy.
    Dataset<FiniteAction, FiniteState>& data = collection.data;

    // Dataset is written into a file
    ofstream out(fm.addPath("dataset.csv"), ios_base::out);
    out << std::setprecision(OS_PRECISION);
    if(out.is_open())
        data.writeToStream(out);
    out.close();

    cout << endl << "# Ended data collection and save" << endl;

    /* The basis functions for the features of the regressor are created here.
     * IdentityBasis functions are features that simply replicate the input. These
     * are useful for the purpose of this test, but very likely we'll need to consider
     * more useful basis functions.
     */
    BasisFunctions bfs = IdentityBasis::generate(2);
    // BasisFunctions qBfs = AndConditionBasisFunction::generate(bfs, 1, nActions);

    // The feature vector is build using the chosen basis functions
    DenseFeatures phi(bfs);

    // The regressor is instantiated using the feature vector
    FFNeuralNetwork nn(phi, 10, 1);


    // ***** NEURAL NETWORK TEST *****
    arma::vec&& rewards = data.rewardAsMatrix();
    arma::mat input = data.featuresAsMatrix(phi);
    arma::rowvec output(input.n_cols, arma::fill::zeros);
    output = rewards.t();
    Q.zeros(input.n_cols);
    nn.getHyperParameters().maxIterations = 10;
    nn.trainFeatures(input, output);
    cout << "The training error is: " << nn.computeJFeatures(input, output, 0) << endl;


    /*
    // A FQI object is instantiated using the dataset and the regressor
    FQI<FiniteState> fqi(data, nn, nActions, 0.9);

    // The FQI procedure starts. It takes the feature vector to be passed to the regressor
    fqi.run(phi, 100, 0.01);
    */

    cout << "END!!!";
    return 0;
}
