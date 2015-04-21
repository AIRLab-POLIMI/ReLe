/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
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

#include "Features.h"
#include "basis/PolynomialFunction.h"
#include "LinearApproximator.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    cout << "## Matrix Test ##" << endl;

    int dim = 1;
    int deg = 5;


    BasisFunctions basis0 = PolynomialFunction::generate(deg, dim);
    BasisFunctions basis2 = PolynomialFunction::generate(deg, dim);
    BasisFunctions basis3 = PolynomialFunction::generate(deg, dim);
    BasisFunctions basis4 = PolynomialFunction::generate(deg, dim);


    cout << endl << "## Sparse features Test ##" << endl;
    SparseFeatures phi0;
    phi0.addBasis(0,0,basis0[2]);
    phi0.addBasis(1,0,basis0[1]);
    phi0.addBasis(2,0,basis0[0]);

    phi0.addBasis(3,1,basis0[3]);
    phi0.addBasis(4,1,basis0[5]);
    phi0.addBasis(5,1,basis0[4]);

    arma::vec pt(1);
    pt[0] = 2;
    cout << "F(" << pt[0] << ") = " << endl;
    cout << phi0(pt) << endl;

    LinearApproximator regressor(phi0);
    arma::vec weights = arma::ones(regressor.getParametersSize());
    regressor.setParameters(weights);
    cout << "W = " << endl << weights << endl;

    cout << "y = F(" << pt[0] << ") * w =" << endl;
    cout << regressor(pt);

    cout << endl << "## Dense features Test 1 (basis) ##" << endl;
    DenseFeatures phi1(new PolynomialFunction(1,5));
    cout << "F(" << pt[0] << ") = " << endl;
    cout << phi1(pt) << endl;

    cout << endl << "## Dense features Test 2 (vector) ##" << endl;
    DenseFeatures phi2(basis2);
    cout << "F(" << pt[0] << ") = " << endl;
    cout << phi2(pt) << endl;

    cout << endl << "## Dense features Test 3 (matrix) ##" << endl;
    DenseFeatures phi3(basis3, 3, 2);
    cout << "F(" << pt[0] << ") = " << endl;
    cout << phi3(pt) << endl;

    cout << endl << "## Dense features Test 4 (matrix) ##" << endl;
    DenseFeatures phi4(basis4, 2, 3);
    cout << "F(" << pt[0] << ") = " << endl;
    cout << phi4(pt) << endl;

    cout << "## Test Polynomial basis ##" << endl;
    arma::vec test(2);
    test(0) = 2;
    test(1) = 6;

    PolynomialFunction basis(2,1);
    cout << "F(" << test.t() << ") = " << endl;
    cout << basis(test) << endl;

    return 0;
}
