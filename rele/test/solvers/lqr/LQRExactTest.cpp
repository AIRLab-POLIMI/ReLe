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

#include "rele/core/Core.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/solvers/lqr/LQRExact.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/SparseFeatures.h"

#include "rele/utils/NumericalGradient.h"

using namespace ReLe;
using namespace std;

int main(int argc, char *argv[])
{
    unsigned int problemsN = 3;
    std::vector<LQR*> problems;
    arma::umat dimensions(problemsN, 2);
    std::vector<arma::vec> controllers;

    //Create LQR problems

    //0
    unsigned int dim0 = 2;
    unsigned int rewardDim0 = 2;
    arma::vec k0 = {-0.4, -0.7};
    LQR* lqr0 = new LQR(dim0, rewardDim0);

    dimensions(0, 0) = dim0;
    dimensions(0, 1) = rewardDim0;
    problems.push_back(lqr0);
    controllers.push_back(k0);


    //1
    unsigned int dim1 = 2;
    unsigned int rewardDim1 = 3;
    arma::vec k1 = {-0.5, -0.36};
    LQR* lqr1 = new LQR(dim1, rewardDim1);

    dimensions(1, 0) = dim1;
    dimensions(1, 1) = rewardDim1;
    problems.push_back(lqr1);
    controllers.push_back(k1);

    //2
    unsigned int dim2 = 2;
    unsigned int rewardDim2 = 2;
    arma::mat A2 = {  {0.1, 0.9}, {0.3, 0.7}};
    arma::mat B2 = {  {0.9, 0.4}, {0.2, 0.5}};
    arma::mat Q2_0(dim2, dim2, arma::fill::randu);
    arma::mat Q2_1(dim2, dim2, arma::fill::randu);
    std::vector<arma::mat> Q2 = {Q2_0, Q2_1};
    arma::mat R2_0(dim2, dim2, arma::fill::randu);
    arma::mat R2_1(dim2, dim2, arma::fill::randu);
    std::vector<arma::mat> R2 = {R2_0, R2_1};
    arma::vec k2 = -arma::vec(dim2, arma::fill::randu);
    LQR* lqr2 = new LQR(A2, B2, Q2, R2);

    dimensions(2, 0) = dim2;
    dimensions(2, 1) = rewardDim2;
    problems.push_back(lqr2);
    controllers.push_back(k2);

    for(unsigned int p = 0; p < problems.size(); p++)
    {
        std::cout << "-----------------------------" << std::endl;
        std::cout << "problem " << p << std::endl;

        //get the problem
        LQR& lqr = *problems[p];
        unsigned int dim = dimensions(p, 0);
        unsigned int rewardDim = dimensions(p, 1);

        //Create LQR Exact
        LQRExact exactLqr(lqr);

        //Setup policy parameters
        arma::mat Sigma = 0.1*arma::eye(dim, dim);
        arma::vec k = controllers[p];

        // Create policy eval agent
        BasisFunctions basis = IdentityBasis::generate(dim);
        SparseFeatures phi;
        phi.setDiagonal(basis);
        MVNPolicy policy(phi, Sigma);
        policy.setParameters(k);
        PolicyEvalAgent<DenseAction, DenseState> agent(policy);

        // Test J
        auto&& core = buildCore(lqr, agent);
        core.getSettings().testEpisodeN = 10000;
        core.getSettings().episodeLength = lqr.getSettings().horizon;
        arma::vec Jsampled = core.runEvaluation();

        arma::vec J = exactLqr.computeJ(k, Sigma);

        std::cout << "Sampled J" << std::endl << Jsampled.t() << std::endl;
        std::cout << "Exact J" << std::endl << J.t() << std::endl;
        std::cout << "Error " << std::endl << arma::abs(Jsampled - J).t() << std::endl;

        // Test gradient
        arma::mat dJ = exactLqr.computeJacobian(k, Sigma);

        auto lambda = [&](const arma::vec& par)
        {
            return exactLqr.computeJ(par, Sigma);
        };

        arma::mat dJnum = NumericalGradient::compute(lambda, k, rewardDim);

        std::cout << "Numerical dJ" << std::endl << dJnum.t() << std::endl;
        std::cout << "Exact dJ" << std::endl << dJ << std::endl;
        std::cout << "Error " << std::endl << arma::abs(dJnum.t() - dJ) << std::endl;


        // Test Hessian
        arma::cube HJ(dim, dim, rewardDim);
        arma::cube HJnum(dim, dim, rewardDim);

        for(unsigned int r = 0; r< rewardDim; r++)
        {
            HJ.slice(r) = exactLqr.computeHesian(k, Sigma, r);

            auto lambda = [&](const arma::vec& par)
            {
                return exactLqr.computeGradient(par, Sigma, r);
            };

            HJnum.slice(r) = NumericalGradient::compute(lambda, k, dim);
        }

        std::cout << "Numerical HJ" << std::endl << HJnum << std::endl;
        std::cout << "Exact HJ" << std::endl << HJ << std::endl;
        std::cout << "Error " << std::endl << arma::abs(HJnum - HJ) << std::endl;
    }

}

