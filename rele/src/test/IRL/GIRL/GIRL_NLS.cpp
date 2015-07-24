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

#include "NLS.h"

#include "PolicyEvalAgent.h"
#include "Core.h"
#include "ParametricRewardMDP.h"
#include "algorithms/GIRL.h"
#include "policy_search/NES/NES.h"
#include "policy_search/gradient/onpolicy/GPOMDPAlgorithm.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IRLGradType atype = IRLGradType::GB;
    int nbEpisodes = 100;

    FileManager fm("lqr", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
    NLS mdp;

    int dim = mdp.getSettings().continuosStateDim;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    BasisFunctions stdBasis = IdentityBasis::generate(dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.5);

    NormalStateDependantStddevPolicy expertPolicy(phi, stdPhi, stdWeights);

    /*** solve the problem in exact way ***/
    arma::vec p(2);
    p(0) = 6.5178;
    p(1) = -2.5994;

    expertPolicy.setParameters(p);

    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    /* Create parametric reward */
    BasisFunctions basisReward = InverseBasis::generate(GaussianRbf::generate({4, 4}, {-10, 10, -10, 10}));
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);


    //Info print
    std::cout << "Basis size: " << phiReward.rows();
    std::cout << " | Params: " << expertPolicy.getParameters().t() << std::endl;

    //Run GIRL
    irlAlg.run();
    arma::vec gnormw = irlAlg.getWeights();

    cout << "Weights (gnorm): " << gnormw.t();


    //Try to recover the initial policy
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;

    NormalStateDependantStddevPolicy imitatorPolicy(phi, stdPhi, stdWeights);
    AdaptiveStep stepRule(0.01);
    int nparams = phi.rows();
    arma::vec mean(nparams, fill::zeros);
    mean[0] = -0.42;
    mean[1] =  0.42;
    /*arma::mat cov(nparams, nparams, arma::fill::eye);
    mat cholMtx = chol(cov);
    ParametricCholeskyNormal dist(mean, cholMtx);
    NES<DenseAction, DenseState> imitator(dist, imitatorPolicy, episodesPerPolicy, policyPerUpdate, stepRule, true);*/

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

    cout << "Learned Parameters: " << imitatorPolicy.getParameters().t() << endl;
    cout << arma::as_scalar(imitatorCore.runBatchTest()) << endl;

    //Evaluate policy against the real mdp
    Core<DenseAction, DenseState> evaluationCore(mdp, imitator);
    evaluationCore.getSettings().loggerStrategy = &emptyStrategy;
    evaluationCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    evaluationCore.getSettings().episodeN = episodes;
    evaluationCore.getSettings().testEpisodeN = nbEpisodes;
    cout << arma::as_scalar(evaluationCore.runBatchTest()) << endl;

    return 0;
}
