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

#include "rele/IRL/algorithms/SCIRL.h"
#include "rele/IRL/algorithms/CSI.h"

using namespace std;
using namespace ReLe;
using namespace arma;

#define TEST

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cout << "choose the algorithm" << std::endl;
		return -1;
	}

	std::string algName(argv[1]);

    FileManager fm("car", algName);
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


    // Create parametric reward
    unsigned tilesRewardN = 15;
    auto* rewardTiles = new BasicTiles({xRange, vRange}, {tilesRewardN, tilesRewardN});
    DenseTilesCoder phiReward(rewardTiles);
    LinearApproximator rewardRegressor(phiReward);

    IRLAlgorithm<FiniteAction, DenseState>* irlAlg;

    if(algName == "scirl")
    {
    	irlAlg = new SCIRL<DenseState>(dataOptimal, rewardRegressor, mdp.getSettings().gamma,
                                         mdp.getSettings().actionsNumber);
    }
    else if(algName == "csi")
    {
    	irlAlg = new CSI<DenseState>(dataOptimal, qphi, rewardRegressor, mdp.getSettings().gamma,
                                       mdp.getSettings().actionsNumber);
    }
    else
    {
    	std::cout << "invalid algorithm choosed" << std::endl;
    	return -1;
    }

    //Run GIRL
    irlAlg->run();
    arma::vec omega = rewardRegressor.getParameters();
    omega.save(fm.addPath("Weights.txt"),  arma::raw_ascii);

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

#ifdef TEST
    imitatorCore.run(1);

    expertPolicy.setEpsilon(0.0);

    core.getSettings().nEpisodes = 1;
    auto&& dataImitator = core.runTest();

    dataImitator.printDecorated(cout);
    cout << "imitator performance: " << dataImitator.getMeanReward(mdp.getSettings().gamma) << endl;
#endif
    return 0;
}
