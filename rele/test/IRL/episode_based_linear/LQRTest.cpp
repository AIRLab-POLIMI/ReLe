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
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/basis/IdentityBasis.h"

#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/LQR.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/utils/FileManager.h"

#include "rele/IRL/algorithms/EMIRL.h"

#include "../RewardBasisLQR.h"
#include "EpisodicCommandLineParser.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    //  RandomGenerator::seed(45423424);
    //  RandomGenerator::seed(8763575);
    CommandLineParser parser;

    auto irlConfig = parser.getConfig(argc, argv);

    int nbEpisodes = irlConfig.episodes;

    FileManager fm("lqr", irlConfig.algorithm);
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    //Set reward policy
    vec eReward = {0.2, 0.7, 0.1};

    // Build policy
    int rewardDim = eReward.n_elem;
    int dim = 3;
    LQR mdp(dim, rewardDim);

    BasisFunctions basis = IdentityBasis::generate(dim);

    SparseFeatures phi;
    phi.setDiagonal(basis);

    DetLinearPolicy<DenseState> expertPolicy(phi);

    /*** solve the problem in exact way ***/
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    arma::mat Sigma = arma::eye(dim, dim);
    Sigma *= 0.001;
    ParametricNormal expertDist(p, Sigma);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertDist.getParameters().t() << std::endl;


    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    /* Create parametric reward */
    BasisFunctions basisReward;
    for(unsigned int i = 0; i < eReward.n_elem; i++)
        basisReward.push_back(new LQR_RewardBasis(i, dim));
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);

    arma::mat theta = expert.getParams();
    auto* irlAlg = buildEpisodicIRLalg(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, irlConfig);



    //Run GIRL
    irlAlg->run();
    arma::vec omega = rewardRegressor.getParameters();

    //Print results
    cout << "Weights: " << omega.t();

    //save stuff
    std::ofstream ofs(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs);
    omega.save(fm.addPath("Weights.txt"),  arma::raw_ascii);
    theta.save(fm.addPath("Theta.txt"),  arma::raw_ascii);

    arma::mat results(rewardRegressor.getParametersSize(), data.size());
    for(unsigned int i = 0; i < data.size(); i++)
    {
        results.col(i) = data[i].computefeatureExpectation(phiReward, mdp.getSettings().gamma);
    }
    results.save(fm.addPath("Phi.txt"),  arma::raw_ascii);

    return 0;
}
