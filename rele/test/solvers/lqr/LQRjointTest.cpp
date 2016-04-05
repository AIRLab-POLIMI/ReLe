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

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/solvers/lqr/LQRExact.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/SparseFeatures.h"

#include "rele/utils/NumericalGradient.h"

#include <nlopt.hpp>

using namespace ReLe;
using namespace std;

arma::vec eReward = { 0.3, 0.7 };
arma::mat Sigma = arma::eye(2, 2)*0.1;

static double objFunctionWrapper(unsigned int n, const double* x, double* grad,
                                 void* o)
{
    arma::vec k(const_cast<double*>(x), n, true);
    LQRExact* lqrExact = static_cast<LQRExact*>(o);

    arma::vec rvec = lqrExact->computeJ(k, Sigma);
    arma::mat dJ = lqrExact->computeJacobian(k, Sigma);

    double value = arma::as_scalar(rvec.t()*eReward);
    arma::vec df = dJ*eReward;

    //Save gradient
    if (grad)
    {
        for (int i = 0; i < df.n_elem; ++i)
        {
            grad[i] = df[i];
        }
    }

    return value;
}

int main(int argc, char *argv[])
{
    //Create LQR problem
    unsigned int dim = 2;
    unsigned int rewardDim = 2;

    LQR lqr(dim, rewardDim);

    //Create LQR Solver
    BasisFunctions basis;
    IdentityBasis* bf1 = new IdentityBasis(0);
    IdentityBasis* bf2 = new IdentityBasis(1);
    basis.push_back(bf1);
    basis.push_back(bf2);

    SparseFeatures phi(basis, 2);
    LQRsolver solver(lqr, phi, LQRsolver::Type::MOO);
    solver.setRewardWeights(eReward);

    arma::mat K = solver.computeOptSolution();
    arma::vec k = K.diag();


    //Create lqr Exact
    LQRExact lqrExact(lqr);

    // optimization
    nlopt::opt optimizator(nlopt::LD_SLSQP, dim);
    optimizator.set_max_objective(objFunctionWrapper, &lqrExact);
    optimizator.set_xtol_rel(1e-8);
    optimizator.set_ftol_rel(1e-8);
    optimizator.set_ftol_abs(1e-8);
    optimizator.set_maxeval(100000);
    optimizator.set_upper_bounds(0.0);
    optimizator.set_lower_bounds(-1.0);

    //optimize dual function
    double minf;
    auto kv = arma::conv_to<std::vector<double>>::from(k);
    if (optimizator.optimize(kv, minf) < 0)
    {
        std::cout << "nlopt failed!" << std::endl;

        return -1;
    }


    std::cout << "found minimum = " << minf << std::endl;
    arma::vec kOpt(kv);
    std::cout << "k solver " << k.t();
    std::cout << "k exact " << kOpt.t();

    arma::mat rvec = lqrExact.computeJ(k, Sigma);
    arma::mat rvecOpt = lqrExact.computeJ(kOpt, Sigma);

    std::cout << "rvec(k)" << rvec.t();
    std::cout << "rvec(kOpt)" << rvecOpt.t();
    std::cout << "J(k) " << arma::as_scalar(rvec.t()*eReward) << std::endl;
    std::cout << "J(kOpt) " << arma::as_scalar(rvecOpt.t()*eReward) << std::endl;

    //Evaluate policies
    BasisFunctions basisPolicy = IdentityBasis::generate(dim);
    SparseFeatures phiPolicy;
    phiPolicy.setDiagonal(basisPolicy);

    MVNPolicy policy(phiPolicy, Sigma);

    PolicyEvalAgent<DenseAction, DenseState> evaluator(policy);

    auto core = buildCore(lqr, evaluator);
    core.getSettings().testEpisodeN = 2000;
    core.getSettings().episodeLength = lqr.getSettings().horizon;

    policy.setParameters(k);
    arma::mat rvecSampled = core.runEvaluation();

    policy.setParameters(kOpt);
    arma::mat rvecOptSampled = core.runEvaluation();

    std::cout << "Estimated parameters " << k.t();
    std::cout << "Optimal parameters " << kOpt.t();

    std::cout << "rvec(k) <sampled>" << rvecSampled.t();
    std::cout << "rvec(kOpt) <sampled>" << rvecOptSampled.t();
    std::cout << "J(k) <sampled>" << arma::as_scalar(rvecSampled.t()*eReward) << std::endl;
    std::cout << "J(kOpt) <sampled>" << arma::as_scalar(rvecOptSampled.t()*eReward) << std::endl;


    // Test gradient
    arma::mat dJ = lqrExact.computeJacobian(k, Sigma);

    auto lambda = [&](const arma::vec& par)
    {
        return lqrExact.computeJ(par, Sigma);
    };

    arma::mat dJnum = NumericalGradient::compute(lambda, k, rewardDim);

    std::cout << "Numerical dJ" << std::endl << dJnum.t() << std::endl;
    std::cout << "Exact dJ" << std::endl << dJ << std::endl;
    std::cout << "Error " << std::endl << arma::abs(dJnum.t() - dJ) << std::endl;


}

