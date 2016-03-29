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
	//Create LQR
	unsigned int dim = 2;
    LQR lqr(dim, dim);

    //Create LQR Exact
    LQRExact exactLqr(lqr);

    //Setup policy parameters
    arma::mat Sigma = arma::eye(dim, dim);
    arma::vec k = {1.0, 1.0};

    // Create policy eval agent
    BasisFunctions basis = IdentityBasis::generate(dim);
    SparseFeatures phi;
    phi.setDiagonal(basis);
    MVNPolicy policy(phi, Sigma);
    policy.setParameters(-k);
    PolicyEvalAgent<DenseAction, DenseState> agent(policy);

    // Test J
    auto&& core = buildCore(lqr, agent);
    core.getSettings().testEpisodeN = 10000;
    core.getSettings().episodeLength = lqr.getSettings().horizon;
    arma::vec Jsampled = core.runEvaluation();

    arma::vec J = exactLqr.computeJ(k, Sigma);

    std::cout << "Sampled J" << std::endl << Jsampled.t() << std::endl;
    std::cout << "Exact J" << std::endl << J.t() << std::endl;

    // Test gradient
    arma::mat dJ = exactLqr.computeJacobian(k, Sigma);

    auto lambda = [&](const arma::vec& par)
    {
        return exactLqr.computeJ(par, Sigma);
    };

    arma::mat dJnum = NumericalGradient::compute(lambda, k, dim);

    std::cout << "Numerical dJ" << std::endl << dJnum << std::endl;
    std::cout << "Exact dJ" << std::endl << dJ << std::endl;


    // Test Hessian
    arma::cube HJ(dim, dim, dim);
    arma::cube HJnum(dim, dim, dim);

    for(unsigned int r = 0; r< dim; r++)
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

}

