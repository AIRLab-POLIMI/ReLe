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

#include "rele/environments/LQR.h"
#include "rele/solvers/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/IRL/algorithms/MGIRL.h"
#include "rele/IRL/algorithms/PGIRL.h"

#include "rele/utils/FileManager.h"

#include "../RewardBasisLQR.h"

using namespace std;
using namespace arma;
using namespace ReLe;


int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IrlGrad atype = IrlGrad::GPOMDP_BASELINE;
    vec eReward = {0.2, 0.7, 0.1};
    int nbEpisodes = 5000;

    FileManager fm("lqr", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
    int dim = eReward.n_elem;
    LQR mdp(dim, dim);

    BasisFunctions basis;
    for (int i = 0; i < dim; ++i)
    {
        basis.push_back(new IdentityBasis(i));
    }

    SparseFeatures phi;
    phi.setDiagonal(basis);

    MVNPolicy expertPolicy(phi);

    /*** solve the problem in exact way ***/
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    expertPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertPolicy.getParameters().t() << std::endl;


    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

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


    LinearApproximator rewardRegressor1(phiReward);
    LinearApproximator rewardRegressor2(phiReward);

    MGIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor1,
                                         mdp.getSettings().gamma, atype);

    PlaneGIRL<DenseAction, DenseState> irlAlg2(data, expertPolicy, rewardRegressor2,
            mdp.getSettings().gamma, atype);


    //Run GIRL
    irlAlg.run();
    arma::vec gnormw = rewardRegressor1.getParameters();

    //Run PGIRL
    irlAlg2.run();
    arma::vec planew = rewardRegressor2.getParameters();


    //Print results
    cout << "Weights (gnorm): " << gnormw.t();
    cout << "Weights (plane): " << planew.t();

    return 0;
}
