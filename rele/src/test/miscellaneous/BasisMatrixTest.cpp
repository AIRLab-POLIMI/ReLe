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

#include "BasisFunctions.h"
#include "LinearApproximator.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    cout << "## Matrix Test ##" << endl;

    int dim = 1;
    int deg = 5;
    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(deg, dim);
    cout << basis << endl;

    SparseBasisMatrix spm;
    spm.addBasis(0,0,basis[2]);
    spm.addBasis(1,0,basis[1]);
    spm.addBasis(2,0,basis[0]);

    spm.addBasis(3,1,basis[3]);
    spm.addBasis(4,1,basis[5]);

    arma::vec pt(1);
    pt[0] = 2;
    cout << "F(" << pt[0] << ") = " << endl;
    cout << spm(pt) << endl;

    LinearApproximator regressor(dim,spm);
    arma::vec weights = arma::ones(regressor.getParametersSize());
    regressor.setParameters(weights);
    cout << "W = " << endl << weights << endl;

    cout << "y = F(" << pt[0] << ") * w =" << endl;
    cout << regressor(pt);


    cout << endl << "## Matrix Test ##" << endl;
    mat evalBasis = basis(pt);
    cout << evalBasis << endl;
    cout << "You see, it is a column vector!!" << endl;

    return 0;
}
