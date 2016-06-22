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

#include "rele/IRL/ParametricRewardMDP.h"

#include "rele/utils/FileManager.h"

#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/TilesCoder.h"
#include "rele/approximators/tiles/BasicTiles.h"

#include "rele/algorithms/batch/td/LSPI.h"

#include "rele/environments/CarOnHill.h"

#include "rele/statistics/DifferentiableNormals.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"

#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/CurvatureEGIRL.h"
#include "rele/IRL/algorithms/SDPEGIRL.h"

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
    CarOnHill mdp;

    // Define linear regressors
    unsigned int tilesN = 15;
    unsigned int actionsN = mdp.getSettings().actionsNumber;
    Range xRange(-1, 1);
    Range vRange(-3, 3);

    auto* tiles = new BasicTiles({xRange, vRange, Range(-0.5, 1.5)}, {tilesN, tilesN, actionsN});

    DenseTilesCoder qphi(tiles);

    LinearApproximator linearQ(qphi);


    //Define solver
    double epsilon = 1e-6;
    LSPI batchAgent(linearQ, epsilon);

    e_GreedyApproximate expertPolicy;
    batchAgent.setPolicy(expertPolicy);

    //Run experiments and learning
    auto&& core = buildBatchCore(mdp, batchAgent);

    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().nEpisodes = 1000;
    core.getSettings().maxBatchIterations = 30;
    core.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("car.log"));
    core.getSettings().agentLogger = new BatchAgentPrintLogger<FiniteAction, DenseState>();

    core.run(1);


    expertPolicy.setEpsilon(0.0);

    core.getSettings().nEpisodes = 1;
    auto&& dataOptimal = core.runTest();

    std::cout << std::endl << "--- Running Test episode ---" << std::endl << std::endl;
    dataOptimal.printDecorated(std::cout);


    //Create expert distribution
    LinearApproximator energyApproximator(qphi);
    GenericParametricGibbsPolicyAllPref<DenseState> policyFamily(mdp.getSettings().actionsNumber, energyApproximator, 1.0);

    vec muExpert = linearQ.getParameters();
    mat SigmaExpert = 1*eye(muExpert.n_elem, muExpert.n_elem);
    ParametricNormal expertDist(muExpert, SigmaExpert);

    //Generate dataset from expert distribution
    PolicyEvalDistribution<FiniteAction, DenseState> expert(expertDist, policyFamily);
    Core<FiniteAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = 1000;
    expertCore.runTestEpisodes();
    Dataset<FiniteAction,DenseState>& data = collection.data;

    // Create parametric reward
    unsigned tilesRewardN = 7;
    auto* rewardTiles = new BasicTiles({xRange, vRange}, {tilesRewardN, tilesRewardN});
    DenseTilesCoder phiReward(rewardTiles);
    LinearApproximator rewardRegressor(phiReward);

    //Create IRL algorithm to run
    arma::mat theta = expert.getParams();
    auto* irlAlg = new EGIRL<FiniteAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);
    /*auto* irlAlg = new CurvatureEGIRL<FiniteAction, DenseState>(data, theta, expertDist,
                rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);*/
    /*auto* irlAlg = new SDPEGIRL<FiniteAction, DenseState>(data, theta, expertDist,
                    rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);*/


    //Run GIRL
    irlAlg->run();
    arma::vec omega = rewardRegressor.getParameters();

    //Learn back environment
    ParametricRewardMDP<FiniteAction, DenseState> prMdp(mdp, rewardRegressor);
    batchAgent.setPolicy(expertPolicy);

    //Run experiments and learning
    expertPolicy.setEpsilon(1.0);
    auto&& imitatorCore = buildBatchCore(prMdp, batchAgent);

    imitatorCore.getSettings().episodeLength = mdp.getSettings().horizon;
    imitatorCore.getSettings().nEpisodes = 1000;
    imitatorCore.getSettings().maxBatchIterations = 30;
    //imitatorCore.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("car.log"));
    imitatorCore.getSettings().agentLogger = new BatchAgentPrintLogger<FiniteAction, DenseState>();

    imitatorCore.run(1);

    expertPolicy.setEpsilon(0.0);

    core.getSettings().nEpisodes = 1;
    auto&& dataImitator = core.runTest();

    dataImitator.printDecorated(cout);
    cout << "imitator performance: " << dataImitator.getMeanReward(mdp.getSettings().gamma) << endl;

    return 0;
}
