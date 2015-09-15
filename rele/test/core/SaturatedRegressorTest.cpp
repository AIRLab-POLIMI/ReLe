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
#include "features/SparseFeatures.h"
#include "regressors/SaturatedRegressor.h"

#include "Utils.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    BasisFunctions basis = IdentityBasis::generate(2);
    SparseFeatures phi(basis, 2);

    arma::vec input1 = {0.0, 0.0};
    arma::vec input2 = {100.0, 100.0};
    arma::vec input3 = {-100.0, -100.0};

    arma::vec w(phi.rows(), arma::fill::ones);

    arma::vec uMin(2, arma::fill::zeros);
    arma::vec uMax(2, arma::fill::ones);
    uMin -= 1;

    SaturatedRegressor regressor1(phi,uMin, uMax);

    regressor1.setParameters(w);

    std::cout << "- Regressor 1" << std::endl;
    std::cout << "eval:" << std::endl;
    std::cout << input1.t() << "-> " << regressor1(input1).t() << std::endl;
    std::cout << input2.t() << "-> " << regressor1(input2).t() << std::endl;
    std::cout << input3.t() << "-> " << regressor1(input3).t() << std::endl;
    std::cout << "diff:" << std::endl;
    std::cout << input1.t() << "-> " << regressor1.diff(input1).t() << std::endl;
    std::cout << input2.t() << "-> " << regressor1.diff(input2).t() << std::endl;
    std::cout << input3.t() << "-> " << regressor1.diff(input3).t() << std::endl;


    uMin = {-1,5};
    uMax = {1, 8};

    SaturatedRegressor regressor2(phi,uMin, uMax);

    regressor2.setParameters(w);

    std::cout << "- Regressor 2:" << std::endl;
    std::cout << "eval:" << std::endl;
    std::cout << input1.t() << "-> " << regressor2(input1).t() << std::endl;
    std::cout << input2.t() << "-> " << regressor2(input2).t() << std::endl;
    std::cout << input3.t() << "-> " << regressor2(input3).t() << std::endl;
    std::cout << "diff:" << std::endl;
    std::cout << input1.t() << "-> " << regressor2.diff(input1).t() << std::endl;
    std::cout << input2.t() << "-> " << regressor2.diff(input2).t() << std::endl;
    std::cout << input3.t() << "-> " << regressor2.diff(input3).t() << std::endl;


}
