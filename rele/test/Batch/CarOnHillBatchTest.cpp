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

#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/utils/FileManager.h"
#include "rele/core/BatchAgent.h"

#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/regressors/trees/ExtraTreeEnsemble.h"
#include "rele/approximators/regressors/trees/KDTree.h"

#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/TilesCoder.h"

#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/approximators/tiles/BasicTiles.h"

#include "rele/environments/CarOnHill.h"
#include "rele/environments/MountainCar.h"

#include "rele/algorithms/batch/td/FQI.h"
#include "rele/algorithms/batch/td/DoubleFQI.h"
#include "rele/algorithms/batch/td/LSPI.h"

using namespace std;
using namespace ReLe;
using namespace arma;

enum alg
{
    fqi = 0, dfqi = 1, lspi = 2
};

int main(int argc, char *argv[])
{
    FileManager fm("car_on_hill", "batch");
    fm.createDir();
    fm.cleanDir();

    // Define domain
    CarOnHill mdp;

    BasisFunctions bfs;
    bfs = IdentityBasis::generate(mdp.getSettings().stateDimensionality + mdp.getSettings().actionDimensionality);

    DenseFeatures phi(bfs);

    // Define tree regressors
    arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> QRegressorA(phi, defaultNode);
    KDTree<arma::vec, arma::vec> QRegressorB(phi, defaultNode);


    // Define linear regressors
    unsigned int tilesN = 25;
    unsigned int actionsN = mdp.getSettings().actionsNumber;
    Range xRange(-1, 1);
    Range vRange(-3, 3);

    auto* tiles = new BasicTiles({xRange, vRange, Range(-0.5, 2.5)},{tilesN, tilesN, actionsN});

    DenseTilesCoder qphi(tiles);

    LinearApproximator linearQ(qphi);

    // Define algorithm
    double epsilon = 1e-6;
    BatchTDAgent<DenseState>* batchAgent;
    alg algorithm = lspi;
    switch(algorithm)
    {
    case fqi:
        batchAgent = new FQI(QRegressorA, epsilon);
        break;
    case dfqi:
        batchAgent = new DoubleFQI(QRegressorA, QRegressorB, epsilon);
        break;
    case lspi:
        batchAgent = new LSPI(linearQ, epsilon);
        break;

    default:
        break;
    }


    //Run experiments and learning
    //*
    auto&& core = buildBatchCore(mdp, *batchAgent);
    core.getSettings().episodeLength = 3000;
    core.getSettings().nEpisodes = 1000;
    core.getSettings().maxBatchIterations = 100;
    core.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("car.log"));
    core.getSettings().agentLogger = new BatchAgentPrintLogger<FiniteAction, DenseState>();

	e_GreedyApproximate policy;
    policy.setEpsilon(1);
    policy.setNactions(mdp.getSettings().actionsNumber);
    core.run(policy);

    /*/

    Dataset<FiniteAction, DenseState> data;
    ifstream ifs("/home/dave/batch.dat");
    data.readFromStream(ifs);
    auto&& core = buildBatchOnlyCore(mdp.getSettings(), data, *batchAgent);
    core.getSettings().maxBatchIterations = 100;
    core.getSettings().logger = new BatchAgentPrintLogger<FiniteAction, DenseState>();
    core.run();

    //*/


    // Policy test
    e_GreedyApproximate epsP;
    epsP.setEpsilon(0.0);
    batchAgent->setPolicy(epsP);
    Policy<FiniteAction, DenseState>* testPolicy = batchAgent->getPolicy();
    PolicyEvalAgent<FiniteAction, DenseState> testAgent(*testPolicy);

    auto&& testCore = buildCore(mdp, testAgent);

    testCore.getSettings().loggerStrategy = new PrintStrategy<FiniteAction, DenseState>();
    testCore.getSettings().episodeLength = 300;
    testCore.getSettings().testEpisodeN = 3;

    testCore.runTestEpisodes();

    return 0;
}
