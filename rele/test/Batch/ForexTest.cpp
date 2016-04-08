/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
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

#include "rele/environments/Forex.h"
#include "rele/core/Core.h"
#include "rele/core/BatchCore.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/algorithms/batch/td/DoubleFQI.h"
#include "rele/algorithms/td/Q-Learning.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/utils/FileManager.h"
#include "rele/core/FiniteMDP.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/regressors/trees/KDTree.h"
#include "rele/algorithms/td/DoubleQ-Learning.h"
#include "rele/algorithms/batch/td/W-FQI.h"

#include <iostream>


using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    // ****** Online ******


    // ******* FQI *******
    BasisFunctions bfs;
    bfs = IdentityBasis::generate(2);

    // The feature vector is build using the chosen basis functions
    DenseFeatures phi(bfs);

    // Tree
    arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> QRegressorA(phi, defaultNode, 1, 1);
    KDTree<arma::vec, arma::vec> QRegressorB(phi, defaultNode, 1, 1);

    Dataset<FiniteAction, FiniteState> data;
    // Dataset is loaded from a file
    ifstream in("/home/shirokuma/Dropbox/Università/Dottorato/Projects/ICML2016/Experiments/Forex/sarsRele.csv", ios_base::in);
    in >> std::setprecision(OS_PRECISION);
    if(in.is_open())
        data.readFromStream(in);
    in.close();

    arma::mat testSet;
    testSet.load(
        "/home/shirokuma/Dropbox/Università/Dottorato/Projects/ICML2016/Experiments/Forex/trainingSet_d_7.txt");

    arma::uvec indicators = {0, 1, 2, 3, 4, 5, 6};
    unsigned int priceCol = 7;
    unsigned int nStates = arma::prod(testSet.row(0));

    FQI<FiniteState> fqi(QRegressorA, 3, 1e-8);
    //DoubleFQI<FiniteState> fqi(data, QRegressorA, QRegressorB, nStates, 3, 1, 1e-8);
    //W_FQI<FiniteState> fqi(data, QRegressorA, nStates, 3, 1, 1e-8);

    auto&& core = buildBatchOnlyCore(data, fqi);

    cout << "Starting FQI..." << endl;

    core.getSettings().maxBatchIterations = 3;

    double gamma = 1.0;
    EnvironmentSettings envSettings;
    envSettings.gamma = gamma;
    core.run(envSettings);

    // Policy Evaluation
    e_Greedy fqiPolicy;
    fqiPolicy.setEpsilon(0);
    //fqiPolicy.setQ(&fqi.getQ());
    PolicyEvalAgent<FiniteAction, FiniteState> evalAgent(fqiPolicy);

    unsigned int nExperiments = 10;
    arma::vec profits(nExperiments, arma::fill::zeros);
    for(unsigned int i = 0; i < nExperiments; i++)
    {
        Forex&& mdpTest = Forex(testSet, indicators, priceCol);

        auto&& coreTest = buildCore(mdpTest, evalAgent);
        coreTest.getSettings().episodeLength = mdpTest.getDataset().n_rows - 1;

        coreTest.runTestEpisode();
        profits(i) = mdpTest.getProfit();
    }

    cout << "PROFIT: " << arma::mean(profits) << endl;
}
