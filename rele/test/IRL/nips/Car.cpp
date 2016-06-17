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


#include "rele/environments/MountainCar.h"
#include "rele/algorithms/td/LinearSARSA.h"

using namespace std;
using namespace ReLe;
using namespace arma;

enum alg
{
    fqi = 0, dfqi = 1, lspi = 2
};

int main(int argc, char *argv[])
{
    FileManager fm("nips", "car");
    fm.createDir();
    fm.cleanDir();

    // Define domain
    MountainCar mdp;

    Tiles* tiles = new BasicTiles({Range(-1, 1), Range(-3, 3), Range(-0.5, 1.5)}, {6, 12, 2});
    DenseTilesCoder phi(tiles);

    ConstantLearningRateDense alpha(0.2);
    e_GreedyApproximate policy;


    LinearGradientSARSA agent(phi, policy, alpha);

    auto&& core = buildCore(mdp, agent);

    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().episodeN = 1000;
    core.getSettings().testEpisodeN = 1000;

    core.runEpisodes();


    PrintStrategy<FiniteAction, DenseState> strategy;
    core.getSettings().loggerStrategy = &strategy;

    core.runTestEpisodes();

    arma::vec J = core.runEvaluation();

    std::cout << J.t() << std::endl;



    /*
    BasisFunctions bfs;
    bfs = IdentityBasis::generate(mdp.getSettings().stateDimensionality + mdp.getSettings().actionDimensionality);
    DenseFeatures phi(bfs);

    // Define tree regressor
    arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> QRegressorA(phi, defaultNode);

    // Define algorithm
    double epsilon = 1e-6;
    BatchTDAgent<DenseState>* batchAgent = new FQI(QRegressorA, epsilon);

    //Learn mountain car
    auto&& core = buildBatchCore(mdp, *batchAgent);
    core.getSettings().episodeLength = 3000;
    core.getSettings().nEpisodes = 1000;
    core.getSettings().maxBatchIterations = 25;
    //core.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("mc.log"));
    core.getSettings().agentLogger = new BatchAgentPrintLogger<FiniteAction, DenseState>();

    e_GreedyApproximate policy;
    policy.setEpsilon(1.0);
    policy.setNactions(mdp.getSettings().actionsNumber);
    core.run(policy);

    // get a dataset
    policy.setEpsilon(0.0);
    batchAgent->setPolicy(policy);
    auto&& expertDataset = core.runTest();

    ofstream fs(fm.addPath("mc.log"));
    expertDataset.writeToStream(fs);

    std::cout << "mean reward: " << expertDataset.getMeanReward(mdp.getSettings().gamma);*/

    return 0;
}
