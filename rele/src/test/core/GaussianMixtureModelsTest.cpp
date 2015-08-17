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

#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"
#include "regressors/GaussianMixtureModels.h"
#include "regressors/LinearApproximator.h"

#include <nlopt.hpp>

#include "Utils.h"

using namespace std;
using namespace ReLe;

double f(unsigned int n, const double* x, double* grad, void* obj)
{
    GaussianRegressor& regressor = *static_cast<GaussianRegressor*>(obj);

    arma::vec input(n);

    for(int i = 0; i < n; i++)
    {
        input(i) = x[i];
    }

    if(grad != nullptr)
    {
        arma::vec g = -regressor.diff(input); // true only by chance... diff is wrt w, not input

        for(int i = 0; i < n; i++)
        {
            grad[i] = g[i];
        }
    }

    double value = arma::as_scalar(regressor(input));

    /* debug */
    std::cout << "v= " << value << " ";

    std::cout << "x= ";
    for(int i = 0; i < n; i++)
    {
        std::cout << x[i] << " ";
    }

    //std::cout << std::endl;

    if(grad)
    {
        std::cout << "g= ";
        for(int i = 0; i < n; i++)
        {
            std::cout << grad[i] << " ";
        }

        std::cout << std::endl;
    }

    return value;
}

int main(int argc, char *argv[])
{
    int size = 2;

    //Test gaussian Regressor
    BasisFunctions basis = IdentityBasis::generate(size);
    DenseFeatures phi(basis);
    GaussianRegressor regressor(phi);
    arma::vec w(size, arma::fill::randn);
    regressor.setParameters(w);

    nlopt::opt optimizator;
    optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, size);
    optimizator.set_max_objective(f, &regressor);
    optimizator.set_xtol_rel(1e-12);
    optimizator.set_ftol_rel(1e-12);
    optimizator.set_maxeval(600);

    std::vector<double> x(size, 0);
    x[0] = -1;
    x[1] = 1;
    double J;

    optimizator.optimize(x, J);

    std::cout << "Jopt = " << as_scalar(regressor(w)) << std::endl;
    std::cout << "J    = " << J << std::endl;
    std::cout << "xopt = " << w[0] << ", " << w[1] << std::endl;
    std::cout << "x    = " << x[0] << ", " << x[1] << std::endl;

    cout << "Parameters" << endl;
    cout << w.t();
    cout << "Numerical gradient" << endl;
    auto lambda = [&](const arma::vec& par)
    {
        regressor.setParameters(par);
        return arma::as_scalar(regressor(w));
    };
    arma::vec numGrad = utils::computeNumericalGradient(lambda, w);
    cout << numGrad.t();
    cout << "Actual gradient" << endl;
    regressor.setParameters(w);
    arma::vec grad1 = regressor.diff(w);
    cout << grad1.t();
    cout << "Error" << endl;
    cout << numGrad.t() - grad1.t();
    cout << "Error Norm" << endl;
    cout << arma::norm(numGrad.t() - grad1.t()) << endl;


    //Test GaussianMixture Regressor
    std::vector<arma::vec> mu;
    mu.push_back({1.0, 1.0});
    mu.push_back({-1.0, -1.0});
    GaussianMixtureRegressor regressor2(phi, mu);

    std::cout << "Test parameters get" << std::endl;
    std::cout << regressor2.getParametersSize() << std::endl;
    std::cout << regressor2.getParameters().t();

    std::cout << "Test regressor value" << std::endl;
    arma::vec in1 = {0, 0};
    arma::vec in2 = {1, 1};
    arma::vec in3 = {-1, -1};
    std::cout << regressor2(in1).t() << "(0.0585)" << std::endl;
    std::cout << regressor2(in2).t() << "(0.0810)" << std::endl;
    std::cout << regressor2(in3).t() << "(0.0810)" << std::endl;

    std::cout << "Test regressor diff" << std::endl;


    cout << "Parameters" << endl;
    arma::vec parGMM = regressor2.getParameters();
    cout << parGMM.t();
    cout << "Numerical gradient" << endl;
    auto lambda2 = [&](const arma::vec& par)
    {
        regressor2.setParameters(par);
        return arma::as_scalar(regressor2(in1));
    };
    arma::vec numGrad2 = utils::computeNumericalGradient(lambda2, parGMM);
    cout << numGrad2.t();
    cout << "Actual gradient" << endl;
    regressor2.setParameters(parGMM);
    arma::vec grad2 = regressor2.diff(in1);
    cout << grad2.t();
    cout << "Error" << endl;
    cout << (numGrad2 - grad2).t();
    cout << "Error Norm" << endl;
    cout << arma::norm(numGrad2 - grad2) << endl;



    std::cout << "Test parameters set" << std::endl;
    arma::vec newPar = regressor2.getParameters();

    newPar(0) = 0.6;
    newPar(1) = 0.4;
    newPar(2) = 1.5;
    newPar(3) = 1.5;
    newPar(9) = 2.0;
    newPar(10) = 1.0;
    newPar(11) = 0.5;

    regressor2.setParameters(newPar);

    std::cout << regressor2.getParametersSize() << std::endl;
    std::cout << newPar.t();
    std::cout << regressor2.getParameters().t();


}
