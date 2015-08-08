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


}
