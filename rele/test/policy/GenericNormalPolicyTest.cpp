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

#include "parametric/differentiable/GenericNormalPolicy.h"
#include "basis/IdentityBasis.h"
#include "features/SparseFeatures.h"
#include "regressors/SaturatedRegressor.h"

#include "NumericalGradient.h"

using namespace ReLe;

int main(int argc, char *argv[])
{
    BasisFunctions basis = IdentityBasis::generate(2);
    SparseFeatures phi;
    phi.setDiagonal(basis);

    arma::vec uMin = {0.0, -1.0};
    arma::vec uMax = {5.0,  1.0};

    arma::vec w = {1.0, 1.0};
    SaturatedRegressor regressor(phi, uMin, uMax);
    regressor.setParameters(w);

    GenericMVNPolicy policy(regressor);

    arma::vec input = mvnrand({0.0, 0.0}, arma::diagmat(arma::vec({10.0, 10.0})));
    arma::vec output = policy(input);
    arma::vec diff = policy.diff(input, output);
    arma::vec numDiff = arma::vectorise(NumericalGradient::compute(policy, policy.getParameters(), input, output));

    std::cout << "input       : " << input.t();
    std::cout << "output      : " << output.t();
    std::cout << "gradient    : " << diff.t();
    std::cout << "num gradient: " << numDiff.t();



}
