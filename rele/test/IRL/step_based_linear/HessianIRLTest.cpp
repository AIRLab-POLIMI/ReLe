/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/IRL/utils/StepBasedHessianCalculatorFactory.h"
#include "rele/IRL/utils/NonlinearHessianFactory.h"

#include "rele/approximators/features/SparseFeatures.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/basis/IdentityBasis.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"

#include "rele/environments/LQR.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/utils/FileManager.h"

#include "../RewardBasisLQR.h"

using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    IrlHess type = IrlHess::REINFORCE_BASELINE;
    //  RandomGenerator::seed(45423424);
    //  RandomGenerator::seed(8763575);

    vec eReward = {0.2, 0.7, 0.1};
    int nbEpisodes = 5000;

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


    LinearApproximator rewardRegressor(phiReward);

    /* select test weights */
    arma::vec wTest = { 0.15, 0.8, 0.05 };
    rewardRegressor.setParameters(wTest);

    std::cout << "Reward test: " << wTest.t();


    /* Create the gradients calculators */
    auto linearHessian = StepBasedHessianCalculatorFactory<DenseAction, DenseState>::build(type, phiReward, data,
                         expertPolicy, mdp.getSettings().gamma);
    auto nonlinearHessian = NonlinearHessianFactory<DenseAction, DenseState>::build(type, rewardRegressor, data,
                            expertPolicy, mdp.getSettings().gamma);


    std::cout << "linear:" << std::endl << linearHessian->computeHessian(wTest);
    std::cout << "linear diff:" << std::endl << linearHessian->getHessianDiff();

    nonlinearHessian->compute();
    std::cout << "non linear:" << std::endl << nonlinearHessian->getHessian();
    std::cout << "non linear diff:" << std::endl << nonlinearHessian->getHessianDiff();

}
