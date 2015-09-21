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
#include "regressors/GaussianMixtureModels.h"
#include "basis/IdentityBasis.h"
#include "basis/PolynomialFunction.h"
#include "basis/GaussianRbf.h"

#include "parametric/differentiable/NormalPolicy.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "DifferentiableNormals.h"

#include "NLS.h"

#include "PolicyEvalAgent.h"
#include "Core.h"
#include "ParametricRewardMDP.h"
#include "algorithms/EMIRL.h"
#include "policy_search/gradient/onpolicy/GPOMDPAlgorithm.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

//#define PRINT
#define RUN_GIRL
#define RECOVER

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    int nbEpisodes = 3000;

    FileManager fm("nls", "EMIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);


    NLS mdp;

    //Setup expert policy
    int dim = mdp.getSettings().continuosStateDim;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    arma::vec p(2);
    p(0) = 6.5178;
    p(1) = -2.5994;

    DetLinearPolicy<DenseState> expertPolicy(phi);
    ParametricFullNormal expertDist(p, 0.1*arma::eye(p.size(), p.size()));

    std::cout << "| Params: " << expertDist.getParameters().t() << std::endl;

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    // Create parametric reward
    BasisFunctions basisReward = GaussianRbf::generate({5, 5}, {-10, 10, -10, 10});

    DenseFeatures phiReward(basisReward);

    LinearApproximator rewardRegressor(phiReward);
    arma::mat theta = expert.getParams();
    EMIRL<DenseAction,DenseState> irlAlg(data, theta, p, arma::eye(p.n_elem, p.n_elem),
                                         phiReward, mdp.getSettings().gamma);

    //Info print
    std::cout << "Basis size: " << phiReward.rows();
    std::cout << " | Params: " << expertPolicy.getParameters().t() << std::endl;
    std::cout << "Features Expectation " << data.computefeatureExpectation(phiReward, mdp.getSettings().gamma).t();

    ofstream ofs2(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs2);


    //Run GIRL
#ifdef RUN_GIRL
    irlAlg.run();
    arma::vec weights = irlAlg.getWeights();
#endif

#ifdef RECOVER

    rewardRegressor.setParameters(weights);

    //Try to recover the initial policy
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;


    BasisFunctions stdBasis = PolynomialFunction::generate(1, dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.1);

    NormalStateDependantStddevPolicy imitatorPolicy(phi, stdPhi, stdWeights);
    AdaptiveStep stepRule(0.01);
    int nparams = phi.rows();
    arma::vec mean(nparams, fill::zeros);

    imitatorPolicy.setParameters(mean);
    GPOMDPAlgorithm<DenseAction, DenseState> imitator(imitatorPolicy, policyPerUpdate,
            mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);

    ParametricRewardMDP<DenseAction, DenseState> prMdp(mdp, rewardRegressor);
    Core<DenseAction, DenseState> imitatorCore(prMdp, imitator);
    EmptyStrategy<DenseAction, DenseState> emptyStrategy;
    imitatorCore.getSettings().loggerStrategy = &emptyStrategy;
    imitatorCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    imitatorCore.getSettings().episodeN = episodes;
    imitatorCore.getSettings().testEpisodeN = nbEpisodes;
    imitatorCore.runEpisodes();

    cout << "----------------------------------------------------------" << endl;
    cout << "Learned Parameters: " << imitatorPolicy.getParameters().t();
    cout << arma::as_scalar(imitatorCore.runBatchTest()) << endl;

    //Evaluate policy against the real mdp
    Core<DenseAction, DenseState> evaluationCore(mdp, imitator);
    CollectorStrategy<DenseAction, DenseState> collector2;
    evaluationCore.getSettings().loggerStrategy = &collector2;
    evaluationCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    evaluationCore.getSettings().episodeN = episodes;
    evaluationCore.getSettings().testEpisodeN = nbEpisodes;

    evaluationCore.runTestEpisodes();

    Dataset<DenseAction,DenseState>& data2 = collector2.data;

    double gamma = mdp.getSettings().gamma;
    cout << "Features Expectation ratio: " << (data2.computefeatureExpectation(phiReward, gamma)/data.computefeatureExpectation(phiReward, gamma)).t();
    cout << "reward: " << arma::as_scalar(evaluationCore.runBatchTest()) << endl;


    stringstream ss;
    ss << "TrajectoriesImitator.txt";
    ofstream ofs3(fm.addPath(ss.str()));
    data2.writeToStream(ofs3);

#endif

    return 0;
}
