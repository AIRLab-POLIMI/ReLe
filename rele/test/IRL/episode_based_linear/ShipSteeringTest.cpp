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

#include "rele/environments/ShipSteering.h"

#include "rele/core/PolicyEvalAgent.h"
#include "rele/core/Core.h"
#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/IRL/algorithms/EMIRL.h"
#include "rele/algorithms/policy_search/gradient/REINFORCEAlgorithm.h"

#include "rele/utils/FileManager.h"

#include "EpisodicCommandLineParser.h"

using namespace std;
using namespace arma;
using namespace ReLe;

#define RUN_GIRL
#define RECOVER

arma::vec learnShipSteering(Environment<DenseAction, DenseState>& mdp, DenseFeatures& phi, int nbEpisodes)
{
    //Learning parameters
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;
    int testEpisodes = 100;
    AdaptiveGradientStep stepRule(0.01);

    int dim = mdp.getSettings().stateDimensionality;

    double epsilon = 0.05;
    NormalPolicy policy(epsilon, phi);

    // Solve the problem with REINFORCE
    REINFORCEAlgorithm<DenseAction, DenseState> expert(policy, policyPerUpdate, stepRule);

    Core<DenseAction, DenseState> expertCore(mdp, expert);
    expertCore.getSettings().loggerStrategy = nullptr;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().episodeN = episodes;
    expertCore.getSettings().testEpisodeN = testEpisodes;
    expertCore.runEpisodes();

    return policy.getParameters();
}

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);
    CommandLineParser parser;

    auto irlConfig = parser.getConfig(argc, argv);

    int nbEpisodes = irlConfig.episodes;

    FileManager fm("ship", irlConfig.algorithm);
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);


    ShipSteering mdp;

    //Setup expert policy
    int dim = mdp.getSettings().stateDimensionality;

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

    arma::vec p = learnShipSteering(mdp, phi, nbEpisodes);

    DetLinearPolicy<DenseState> expertPolicy(phi);
    ParametricNormal expertDist(p, 0.1*arma::eye(p.size(), p.size()));

    std::cout << "Params: " << expertDist.getParameters().t() << std::endl;

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    /* Generate expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    // Create parametric reward
    BasisFunctions basisReward = GaussianRbf::generate({20, 20}, {0, 150, 0, 150});

    DenseFeatures phiReward(basisReward);

    LinearApproximator rewardRegressor(phiReward);
    arma::mat theta = expert.getParams();
    auto* irlAlg = buildEpisodicIRLalg(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, irlConfig);

    //Info print
    std::cout << "Basis size: " << phiReward.rows();
    std::cout << " | Params: " << expertPolicy.getParameters().t() << std::endl;
    std::cout << "Features Expectation " << data.computefeatureExpectation(phiReward, mdp.getSettings().gamma).t();

    ofstream ofs1(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs1);


    /* RUN */
    irlAlg->run();
    arma::vec weights = rewardRegressor.getParameters();


    /* RECOVER! */

    //Try to recover the initial policy
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;


    BasisFunctions stdBasis = PolynomialFunction::generate(1, dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.1);

    ParametricRewardMDP<DenseAction, DenseState> prMdp(mdp, rewardRegressor);
    arma::vec mean = learnShipSteering(prMdp, phi, nbEpisodes);;

    double epsilon = 0.05;
    NormalPolicy imitatorPolicy(epsilon, phi);
    imitatorPolicy.setParameters(mean);

    cout << "----------------------------------------------------------" << endl;
    cout << "Learned Parameters: " << imitatorPolicy.getParameters().t();

    //Evaluate policy against the real mdp
    PolicyEvalAgent<DenseAction, DenseState> imitator(imitatorPolicy);
    Core<DenseAction, DenseState> evaluationCore(mdp, imitator);
    CollectorStrategy<DenseAction, DenseState> collector2;
    evaluationCore.getSettings().loggerStrategy = &collector2;
    evaluationCore.getSettings().episodeLength = mdp.getSettings().horizon;
    evaluationCore.getSettings().episodeN = episodes;
    evaluationCore.getSettings().testEpisodeN = nbEpisodes;

    evaluationCore.runTestEpisodes();

    Dataset<DenseAction,DenseState>& data2 = collector2.data;

    double gamma = mdp.getSettings().gamma;
    cout << "Features Expectation ratio: " << (data2.computefeatureExpectation(phiReward, gamma)/data.computefeatureExpectation(phiReward, gamma)).t();
    cout << "reward: " << arma::as_scalar(evaluationCore.runEvaluation()) << endl;

    ofstream ofs2(fm.addPath("TrajectoriesImitator.txt"));
    data2.writeToStream(ofs2);

    return 0;
}
