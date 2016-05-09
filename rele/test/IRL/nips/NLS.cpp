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

#include "rele/approximators/features/SparseFeatures.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/GaussianMixtureModels.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/NLS.h"

#include "rele/core/PolicyEvalAgent.h"
#include "rele/core/Core.h"
#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/SDPEGIRL.h"
#include "rele/IRL/algorithms/CurvatureEGIRL.h"
#include "rele/algorithms/policy_search/gradient/GPOMDPAlgorithm.h"

#include "rele/utils/FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Wrong argument number: n_episode, n_experiment must be provided" << endl;
        return -1;
    }

    string n_episodes(argv[1]);
    string n_experiment(argv[2]);

    FileManager fm("nips/nls_exact/" + n_episodes + "/" + n_experiment);
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    unsigned int nbEpisodes = stoi(n_episodes);

    // Create parametric reward MDP
    BasisFunctions basisReward = GaussianRbf::generate({5, 5}, {-2, 2, -2, 2});
    DenseFeatures phiReward(basisReward);
    LinearApproximator rewardRegressor(phiReward);
    arma::vec v(phiReward.rows(), arma::fill::zeros);
    v(12) = 1.0;
    rewardRegressor.setParameters(v);

    NLS mdp;
    ParametricRewardMDP<DenseAction, DenseState> prMDP(mdp, rewardRegressor);

    //Setup expert policy
    int dim = mdp.getSettings().stateDimensionality;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    arma::vec p = {11.5431, -3.8302};

    DetLinearPolicy<DenseState> expertPolicy(phi);

    /*ParametricNormal dist(p, arma::eye(p.n_elem, p.n_elem));

    AdaptiveGradientStep step(0.01);
    PGPE<DenseAction, DenseState> agent(dist,expertPolicy, 1, 100, step);

    auto core = buildCore(prMDP, agent);
    CollectorStrategy<DenseAction, DenseState> collector;
    core.getSettings().loggerStrategy = &collector;
    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().episodeN = 10000;
    core.getSettings().testEpisodeN = 100;

    core.runEpisodes();


    auto& data = collector.data;
    double gamma = mdp.getSettings().gamma;
    cout << "Features Expectation ratio: " << data.computefeatureExpectation(phiReward, gamma).t();
    cout << "reward: " << core.runEvaluation() << endl;


    stringstream ss;
    ss << "Trajectories.txt";
    ofstream ofs(fm.addPath(ss.str()));
    data.writeToStream(ofs);

    std::cout << dist.getParameters().t() << std::endl;*/

    ParametricNormal expertDist(p, 0.1*arma::eye(p.size(), p.size()));

    std::cout << "Distribution Params: " << expertDist.getParameters().t() << std::endl;

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    // Generate expert dataset
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    // Learn reward weights
    arma::mat theta = expert.getParams();

    auto* irlAlg1 = new EGIRL<DenseAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);
    irlAlg1->run();
    arma::vec weights1 = rewardRegressor.getParameters();

    auto* irlAlg2 = new SDPEGIRL<DenseAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);
    irlAlg2->run();
    arma::vec weights2 = rewardRegressor.getParameters();

    auto* irlAlg3 = new CurvatureEGIRL<DenseAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);
    irlAlg3->run();
    arma::vec weights3 = rewardRegressor.getParameters();

    //plot results
    std::cout << "EGIRL: " << weights1(12) << std::endl;
    std::cout << "SDPEGIRL: " << weights2(12) << std::endl;
    std::cout << "CurvatureEGIRL: " << weights3(12) << std::endl;

    // Save Reward Function
    weights1.save(fm.addPath("Weights1.txt"), arma::raw_ascii);
    weights2.save(fm.addPath("Weights2.txt"), arma::raw_ascii);
    weights3.save(fm.addPath("Weights3.txt"), arma::raw_ascii);

    return 0;
}
