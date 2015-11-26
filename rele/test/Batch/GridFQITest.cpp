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
#include "BasisFunctions.h"
#include "features/DenseFeatures.h"
#include "FileManager.h"
#include "FiniteMDP.h"
#include "GridWorldGenerator.h"
#include "regressors/FFNeuralNetwork.h"
#include "basis/IdentityBasis.h"

#include <iostream>
#include <string>

using namespace std;
using namespace ReLe;
using namespace arma;


int main(int argc, char *argv[])
{
    FileManager fm("gw", "FQI");
    fm.createDir();
    fm.cleanDir();

    unsigned int nbEpisodes = 5000;

    GridWorldGenerator generator;
    generator.load(argv[1]);

    FiniteMDP&& mdp = generator.getMDP(1.0);

    vector<FiniteAction> actions;
    for (unsigned int i = 0; i < mdp.getSettings().finiteActionDim; i++)
        actions.push_back(FiniteAction(i));

    //--- policy setup
    e_Greedy randomPolicy;
    randomPolicy.setEpsilon(1.);
    PolicyEvalAgent<FiniteAction, FiniteState> expert(randomPolicy);

    Core<FiniteAction, FiniteState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, FiniteState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    unsigned int horizon = mdp.getSettings().horizon;
    expertCore.getSettings().episodeLenght = horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<FiniteAction, FiniteState>& data = collection.data;

    ofstream out(fm.addPath("Dataset.csv"), ios_base::out);
    if(out.is_open())
        data.writeToStream(out);
    out.close();

    cout << "# Ended data collection and save" << endl;

    unsigned int nActions = actions.size();
    BasisFunctions bfs = IdentityBasis::generate(2); // FIXME
    DenseFeatures phi(bfs);

    FFNeuralNetwork nn(phi, 10, 1);
    FQI<FiniteState> fqi(data, nn, nActions, 0.9);
    fqi.run(phi, 100, 0.01);

    return 0;
}
