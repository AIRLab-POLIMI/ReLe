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

#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"

#include "rele/environments/NLS.h"

#include "rele/core/PolicyEvalAgent.h"
#include "rele/core/Core.h"
#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/algorithms/policy_search/gradient/GPOMDPAlgorithm.h"

#include "rele/utils/FileManager.h"

#include "StepBasedCommandLineParser.h"


using namespace std;
using namespace arma;
using namespace ReLe;


int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);
    CommandLineParser parser;

    auto irlConfig = parser.getConfig(argc, argv);


    IrlGrad atype = irlConfig.gradient;
    int nbEpisodes = irlConfig.episodes;

    FileManager fm("nls", irlConfig.algorithm);
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);


    NLS mdp;

    //Setup expert policy
    int dim = mdp.getSettings().stateDimensionality;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    BasisFunctions stdBasis = PolynomialFunction::generate(1, dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.1);

    NormalStateDependantStddevPolicy expertPolicy(phi, stdPhi, stdWeights);

    arma::vec p(2);
    p(0) = 6.5178;
    p(1) = -2.5994;

    expertPolicy.setParameters(p);

    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    // Generate expert dataset
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    // Create parametric reward
    BasisFunctions basisReward = GaussianRbf::generate({5, 5}, {-2, 2, -2, 2});
    DenseFeatures phiReward(basisReward);
    LinearApproximator rewardRegressor(phiReward);

    //Info print
    std::cout << "Reward Basis size: " << phiReward.rows() << std::endl;
    std::cout << "Policy Params size: " << expertPolicy.getParametersSize() << std::endl;
    std::cout << "Policy Params: " << expertPolicy.getParameters().t();
    std::cout << "Features Expectation " << data.computefeatureExpectation(phiReward, mdp.getSettings().gamma).t();

    ofstream ofs(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs);


    //Run algorithm
    IRLAlgorithm<DenseAction,DenseState>* irlAlg = buildIRLalg(data, expertPolicy, rewardRegressor,
            mdp.getSettings().gamma, irlConfig);

    irlAlg->run();
    arma::vec weights = rewardRegressor.getParameters();

    cout << "weights :" << weights.t();

    rewardRegressor.setParameters(weights);

    //Try to recover the initial policy
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;

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
    imitatorCore.getSettings().episodeN = episodes;
    imitatorCore.getSettings().testEpisodeN = nbEpisodes;
    imitatorCore.runEpisodes();

    cout << "----------------------------------------------------------" << endl;
    cout << "Learned Parameters: " << imitatorPolicy.getParameters().t();
    cout << arma::as_scalar(imitatorCore.runEvaluation()) << endl;

    //Evaluate policy against the real mdp
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
    cout << "reward: " << arma::as_scalar(data2.getMeanReward(gamma)) << endl;

    ofstream ofs2(fm.addPath("TrajectoriesImitator.txt"));
    data2.writeToStream(ofs2);


    // Save Reward Function
    weights.save(fm.addPath("Weights.txt"),  arma::raw_ascii);

    return 0;
}




