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
#include "algorithms/GIRL.h"

#include "policy_search/gradient/onpolicy/REINFORCEAlgorithm.h"

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

    IRLGradType atype = IRLGradType::RB;
    int nbEpisodes = 3000;

    //Learning parameters
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;
    int testEpisodes = 100;
    AdaptiveStep stepRule(0.01);

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

    // Solve the problem with REINFORCE
    REINFORCEAlgorithm<DenseAction, DenseState> expert(expertPolicy, policyPerUpdate, stepRule);

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
    BasisFunctions basisReward = GaussianRbf::generate({20, 20}, {0, 150, 0, 150});
    DenseFeatures phiReward(basisReward);

    LinearApproximator rewardRegressor(phiReward);
    GIRL<DenseAction,DenseState> irlAlg1(data, expertPolicy, rewardRegressor,
                                         mdp.getSettings().gamma, atype, None);
    GIRL<DenseAction,DenseState> irlAlg2(data, expertPolicy, rewardRegressor,
                                         mdp.getSettings().gamma, atype, LogDisparity);

    //Info print
    std::cout << "Basis size: " << phiReward.rows();
    std::cout << " | Params: " << expertPolicy.getParameters().t() << std::endl;
    std::cout << "Features Expectation " << data.computefeatureExpectation(phiReward, mdp.getSettings().gamma).t();

    ofstream ofs(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs);


    //Run GIRL
#ifdef RUN_GIRL
    irlAlg1.run();
    arma::vec weights1 = irlAlg1.getWeights();

    irlAlg2.run();
    arma::vec weights2 = irlAlg2.getWeights();


    cout << "weights         (None): " << weights1.t();
    cout << "weights (LogDisparity): " << weights2.t();

    arma::mat weights(weights1.n_rows, 2);

    weights.col(0) = weights1;
    weights.col(1) = weights2;

#endif

#ifdef RECOVER

    for(unsigned int i = 0; i < weights.n_cols; i++)
    {
        rewardRegressor.setParameters(weights.col(i));

        //Try to recover the initial policy
        int episodesPerPolicy = 1;
        int policyPerUpdate = 100;
        int updates = 400;
        int episodes = episodesPerPolicy*policyPerUpdate*updates;

        NormalPolicy imitatorPolicy(epsilon, phi);
        AdaptiveStep stepRule(0.01);
        int nparams = phi.rows();
        arma::vec mean(nparams, fill::zeros);

        imitatorPolicy.setParameters(mean);
        REINFORCEAlgorithm<DenseAction, DenseState> imitator(imitatorPolicy, policyPerUpdate, stepRule);

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
        ss << "TrajectoriesImitator" << i << ".txt";
        ofstream ofs(fm.addPath(ss.str()));
        data2.writeToStream(ofs);
    }
#endif

    return 0;
}
