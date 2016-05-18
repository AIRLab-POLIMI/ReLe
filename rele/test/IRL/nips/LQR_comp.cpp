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

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/LQR.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/utils/FileManager.h"

#include "rele/IRL/algorithms/GIRL.h"

#include "../RewardBasisLQR.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        cout << "Wrong argument number: dimension, n_episode, n_experiment must be provided" << endl;
        return -1;
    }

    string dimension(argv[1]);
    string n_episodes(argv[2]);
    string n_experiment(argv[3]);

    FileManager fm("nips/lqr_comp/" + dimension + "/" + n_episodes + "/" + n_experiment);
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    unsigned int dim = stoi(dimension);
    unsigned int nbEpisodes = stoi(n_episodes);

    //Set reward policy
    arma::vec eReward;

    if(dim == 2)
    {
        eReward = {0.3, 0.7};
    }
    else
    {
        eReward = {0.2, 0.7, 0.1};
    }

    int rewardDim = dim;
    LQR mdp(dim, rewardDim);

    BasisFunctions basis = IdentityBasis::generate(dim);

    SparseFeatures phi;
    phi.setDiagonal(basis);

    // solve the problem in exact way
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    arma::mat Sigma = arma::eye(dim, dim)*5e-2;

    MVNPolicy expertPolicy(phi, Sigma);
    expertPolicy.setParameters(p);


    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    // Generate LQR expert dataset
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    // Create parametric reward
    BasisFunctions basisReward;
    for(unsigned int i = 0; i < eReward.n_elem; i++)
        basisReward.push_back(new LQR_RewardBasis(i, dim));
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);

    auto* irlAlg = new GIRL<DenseAction, DenseState>(data, expertPolicy, rewardRegressor,
            mdp.getSettings().gamma, IrlGrad::GPOMDP_BASELINE);

    //Run GIRL
    irlAlg->run();
    arma::vec omega = rewardRegressor.getParameters();

    //Print results
    cout << "Weights: " << omega.t();

    //save weights
    omega.save(fm.addPath("Weights.txt"),  arma::raw_ascii);

    return 0;
}
