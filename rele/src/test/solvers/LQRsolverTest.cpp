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

#include "LQRsolver.h"
#include "basis/IdentityBasis.h"

using namespace ReLe;
using namespace std;

int main(int argc, char *argv[])
{
    LQR lqr(2,1);

    DenseBasisMatrix basis;
    IdentityBasis* bf1 = new IdentityBasis(0);
    IdentityBasis* bf2 = new IdentityBasis(1);
    basis.push_back(bf1);
    basis.push_back(bf2);

    SparseBasisMatrix basisMatrix(basis, 2);
    LinearApproximator regressor(basis.size(), basisMatrix);

    LQRsolver solver(lqr, regressor);

    solver.solve();


    DetLinearPolicy<DenseState>& policy = static_cast<DetLinearPolicy<DenseState>&>(solver.getPolicy());

    cout << "Optimal Policy:" << endl;
    cout << policy.getParameters() << endl;


    auto&& data = solver.test();

    cout << "Final state:" << endl;
    cout << data.back().back().xn << endl;

}

