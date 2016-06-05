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

#include "rele/core/Core.h"
#include "rele/core/BatchCore.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/utils/FileManager.h"
#include "rele/core/BatchAgent.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/regressors/trees/ExtraTree.h"
#include "rele/approximators/regressors/trees/KDTree.h"
#include "rele/core/logger/BatchDatasetLogger.h"

#include "rele/environments/MountainCar.h"
#include "rele/algorithms/batch/td/FQI.h"
#include "rele/algorithms/batch/td/DoubleFQI.h"

using namespace std;
using namespace ReLe;
using namespace arma;

enum alg
{
    fqi = 0, dfqi = 1, lspi = 2
};

int main(int argc, char *argv[])
{
    // Define domain
    MountainCar mdp(MountainCar::ConfigurationsLabel::Ernst);

    // Define basis
    BasisFunctions bfs;
    bfs = IdentityBasis::generate(mdp.getSettings().stateDimensionality + mdp.getSettings().actionDimensionality);

    DenseFeatures phi(bfs);

    // Define regressors
    arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    ExtraTree<arma::vec, arma::vec> QRegressorA(phi, defaultNode);
    ExtraTree<arma::vec, arma::vec> QRegressorB(phi, defaultNode);

    double epsilon = 1e-8;
    BatchTDAgent<DenseState>* batchAgent;
    alg algorithm = fqi;
    switch(algorithm)
    {
    case fqi:
        batchAgent = new FQI(QRegressorA, epsilon);
        break;
    case dfqi:
        batchAgent = new DoubleFQI(QRegressorA, QRegressorB, epsilon);
        break;
    case lspi:

        break;

    default:
        break;
    }

    auto&& core = buildBatchCore(mdp, *batchAgent);
    core.getSettings().episodeLength = 3000;
    core.getSettings().nEpisodes = 1000;
    core.getSettings().maxBatchIterations = 20;
    FileManager fm("mc", "fqi");
    fm.createDir();
    fm.cleanDir();
    core.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("mc.log"));

    e_GreedyApproximate policy;
    policy.setEpsilon(1);
    policy.setNactions(mdp.getSettings().actionsNumber);
    core.run(policy);

    // Policy test
    e_GreedyApproximate epsP;
    batchAgent->setPolicy(epsP);
    Policy<FiniteAction, DenseState>* testPolicy = batchAgent->getPolicy();
    PolicyEvalAgent<FiniteAction, DenseState> testAgent(*testPolicy);

    auto&& testCore = buildCore(mdp, testAgent);

    testCore.getSettings().loggerStrategy = new PrintStrategy<FiniteAction, DenseState>();
    testCore.getSettings().episodeLength = 300;
    testCore.getSettings().testEpisodeN = 1;

    testCore.runTestEpisodes();

    return 0;
}
