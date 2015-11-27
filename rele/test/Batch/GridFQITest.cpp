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

    // The actions vector is generated considering the number of actions of the MDP
    vector<FiniteAction> actions;
    for (unsigned int i = 0; i < mdp.getSettings().finiteActionDim; i++)
        actions.push_back(FiniteAction(i));

    /* This policy is used by an evaluation agent that has the purpose to
    *  build a dataset from its exploration of the environment. The policy is
    *  a random policy, thus allowing pure exploration. */
    e_Greedy randomPolicy;
    randomPolicy.setEpsilon(1.);
    // The agent is instatiated. It takes the policy as parameter.
    PolicyEvalAgent<FiniteAction, FiniteState> expert(randomPolicy);

    /* The Core class is what ReLe uses to move the agent in the MDP. It is
    *  instatiated using the MDP and the agent itself. */
    Core<FiniteAction, FiniteState> expertCore(mdp, expert);
    /* The CollectorStrategy is used to collect data from the agent that is
    *  moving in the MDP. Here, it is used to store the transition that are used
    *  as the inputs of the dataset provided to FQI. */
    CollectorStrategy<FiniteAction, FiniteState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    // Number of transitions for each episode is retrieved from the MDP settings.
    unsigned int nTransitions = mdp.getSettings().horizon;
    // Number of transitions to be performed is set in the agent settings.
    expertCore.getSettings().episodeLength = nTransitions;
    // Same as before, for episodes
    unsigned int nEpisodes = 5000;
    expertCore.getSettings().testEpisodeN = nEpisodes;
    /* The agent start the exploration that will last for the provided number of
    *  episodes. */
    expertCore.runTestEpisodes();
    // The dataset is build from the data collected by the CollectorStrategy.
    Dataset<FiniteAction, FiniteState>& data = collection.data;

    // Dataset is written into a file
    ofstream out(fm.addPath("Dataset.csv"), ios_base::out);
    if(out.is_open())
        data.writeToStream(out);
    out.close();

    cout << "# Ended data collection and save" << endl;

    /* The basis functions for the features of the regressor are created here.
    *  Using IdentityBasis function, we simply say that the input of the regressor,
    *  are the states and actions values themselves. */
    BasisFunctions bfs = IdentityBasis::generate(2); // FIXME
    // The feature vector is build using the chosen basis functions
    DenseFeatures phi(bfs);

    // The regressor is instatiated using the feature vector
    FFNeuralNetwork nn(phi, 10, 1);
    // FQI needs to know the cardinality of the actions set
    unsigned int nActions = actions.size();
    // A fqi object is instatiated using the dataset and the regressor
    FQI<FiniteState> fqi(data, nn, nActions, 0.9);
    /* The fqi procedure starts. It takes the feature vector to be passed to the
    *  regressor. */
    fqi.run(phi, 100, 0.01);

    return 0;
}
