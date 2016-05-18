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

#include "rele/algorithms/policy_search/NES/NES.h"

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

    FileManager fm("nips/nls_fe/" + n_episodes + "/" + n_experiment);
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
    //ParametricRewardMDP<DenseAction, DenseState> prMDP(mdp, rewardRegressor);

    //Setup expert policy
    int dim = mdp.getSettings().stateDimensionality;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    arma::vec p(2);
    p(0) = 6.5178;
    p(1) = -2.5994;

    DetLinearPolicy<DenseState> expertPolicy(phi);

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

    auto* irlAlg1 = new SDPEGIRL<DenseAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);
    irlAlg1->run();
    arma::vec weights1 = rewardRegressor.getParameters();

    auto* irlAlg2 = new CurvatureEGIRL<DenseAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);
    irlAlg2->run();
    arma::vec weights2 = rewardRegressor.getParameters();

    arma::mat weights(rewardRegressor.getParametersSize(), 2);

    weights.col(0) = weights1;
    weights.col(1) = weights2;

    double gamma = mdp.getSettings().gamma;

    for(unsigned int i = 0; i < 2; i++)
    {
        std::cout << "Learning reward function " << i << std::endl;

        //Set parameters of reward function
        rewardRegressor.setParameters(weights.col(i));

        //Try to recover the initial policy
        int episodesPerPolicy = 1;
        int policyPerUpdate = 100;
        int updates = 400;
        int learningEpisodes = episodesPerPolicy*policyPerUpdate*updates;
        unsigned int testEpisodes = 2000;

        BasisFunctions stdBasis = PolynomialFunction::generate(1, dim);
        DenseFeatures stdPhi(stdBasis);
        arma::vec stdWeights(stdPhi.rows());
        stdWeights.fill(0.1);

        NormalStateDependantStddevPolicy imitatorPolicy(phi, stdPhi, stdWeights);
        AdaptiveGradientStep stepRule(0.01);
        int nparams = phi.rows();
        arma::vec mean(nparams, fill::zeros);

        imitatorPolicy.setParameters(mean);
        GPOMDPAlgorithm<DenseAction, DenseState> imitator(imitatorPolicy, policyPerUpdate,
                mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);

        ParametricRewardMDP<DenseAction, DenseState> prMdp(mdp, rewardRegressor);
        Core<DenseAction, DenseState> imitatorCore(prMdp, imitator);
        imitatorCore.getSettings().loggerStrategy = nullptr;
        imitatorCore.getSettings().episodeLength = mdp.getSettings().horizon;
        imitatorCore.getSettings().episodeN = learningEpisodes;
        imitatorCore.getSettings().testEpisodeN = testEpisodes;
        imitatorCore.runEpisodes();

        //Evaluate policy against the real mdp
        Core<DenseAction, DenseState> evaluationCore(mdp, imitator);
        CollectorStrategy<DenseAction, DenseState> collector2;
        evaluationCore.getSettings().loggerStrategy = &collector2;
        evaluationCore.getSettings().episodeLength = mdp.getSettings().horizon;
        evaluationCore.getSettings().episodeN = learningEpisodes;
        evaluationCore.getSettings().testEpisodeN = nbEpisodes;

        evaluationCore.runTestEpisodes();

        Dataset<DenseAction,DenseState>& data2 = collector2.data;


        // Get results
        arma::vec featureExpectationImitator = data2.computefeatureExpectation(phiReward, gamma);
        arma::vec rewardRealMdp = data2.getMeanReward(gamma);

        featureExpectationImitator.save(fm.addPath("FE_" + std::to_string(i) + ".txt"), arma::raw_ascii);
        rewardRealMdp.save(fm.addPath("R_" + std::to_string(i) + ".txt"), arma::raw_ascii);
    }

    arma::vec featureExpectationExpert = data.computefeatureExpectation(phiReward, gamma);
    arma::vec rewardExpert = data.getMeanReward(gamma);
    featureExpectationExpert.save(fm.addPath("FE_expert.txt"), arma::raw_ascii);
    rewardExpert.save(fm.addPath("R_expert.txt"), arma::raw_ascii);

    return 0;
}
