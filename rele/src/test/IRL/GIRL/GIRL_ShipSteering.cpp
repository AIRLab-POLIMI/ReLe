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

#include "features/SparseFeatures.h"
#include "features/DenseFeatures.h"
#include "regressors/LinearApproximator.h"
#include "basis/IdentityBasis.h"
#include "basis/PolynomialFunction.h"
#include "basis/GaussianRbf.h"

#include "parametric/differentiable/NormalPolicy.h"

#include "ShipSteering.h"

#include "PolicyEvalAgent.h"
#include "Core.h"
#include "ParametricRewardMDP.h"
//#include "algorithms/GIRL.h"
#include "algorithms/PGIRL.h"
#include "policy_search/gradient/onpolicy/GPOMDPAlgorithm.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

	//Learning parameters
	int episodesPerPolicy = 1;
	int policyPerUpdate = 100;
	int updates = 400;
	int episodes = episodesPerPolicy*policyPerUpdate*updates;
	int testEpisodes = 100;
	AdaptiveStep stepRule(0.01);
    IRLGradType atype = IRLGradType::GB;

    FileManager fm("ship", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    //Empty strategy
    EmptyStrategy<DenseAction, DenseState> empty;

    // Learn Ship correct policy
    ShipSteering mdp;

    int dim = mdp.getSettings().continuosStateDim;

    BasisFunctions basis = GaussianRbf::generate(
    {
        3,
        3,
        6,
        2
    },
    {
        0.0, 150.0,
        0.0, 150.0,
        -M_PI, M_PI,
        -15.0, 15.0
    });

    DenseFeatures phi(basis);

    double epsilon = 0.05;
    NormalPolicy expertPolicy(epsilon, phi);

    // Solve the problem with GPOMDP
    GPOMDPAlgorithm<DenseAction, DenseState> expert(expertPolicy, policyPerUpdate,
                mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);

    Core<DenseAction, DenseState> expertCore(mdp, expert);
    expertCore.getSettings().loggerStrategy = &empty;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().episodeN = episodes;
    expertCore.getSettings().testEpisodeN = testEpisodes;
    expertCore.runEpisodes();


    // Generate expert dataset
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    // Create parametric reward
    BasisFunctions basisReward = GaussianRbf::generate({3, 3}, {0, 150, 0, 150});
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);
    PlaneGIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, basisReward,
                                        mdp.getSettings().gamma, atype);


    //Info print
    std::cout << "Basis size: " << phiReward.rows();
    std::cout << " | Params: " << expertPolicy.getParameters().t() << std::endl;

    //Run PGIRL
    irlAlg.run();
    arma::vec weights = irlAlg.getWeights();

    cout << "Weights: " << weights.t();
    rewardRegressor.setParameters(weights);


    //Try to recover the initial policy
    NormalPolicy imitatorPolicy(epsilon, phi);
    int nparams = phi.rows();

    GPOMDPAlgorithm<DenseAction, DenseState> imitator(imitatorPolicy, policyPerUpdate,
            mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);


    ParametricRewardMDP<DenseAction, DenseState> prMdp(mdp, rewardRegressor);
    Core<DenseAction, DenseState> imitatorCore(prMdp, imitator);
    imitatorCore.getSettings().loggerStrategy = &empty;
    imitatorCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    imitatorCore.getSettings().episodeN = episodes;
    imitatorCore.getSettings().testEpisodeN = testEpisodes;
    imitatorCore.runEpisodes();

    cout << "Learned Parameters: " << imitatorPolicy.getParameters().t() << endl;
    cout << arma::as_scalar(imitatorCore.runBatchTest()) << endl;

    //Evaluate policy against the real mdp
    Core<DenseAction, DenseState> evaluationCore(mdp, imitator);
    evaluationCore.getSettings().loggerStrategy = &empty;
    evaluationCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    evaluationCore.getSettings().episodeN = episodes;
    evaluationCore.getSettings().testEpisodeN = testEpisodes;
    cout << arma::as_scalar(evaluationCore.runBatchTest()) << endl;

    return 0;
}
