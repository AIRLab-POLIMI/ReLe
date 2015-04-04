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
#include "basis/PolynomialFunction.h"

using namespace ReLe;

int main(int argc, char *argv[])
{
    LQR lqr(1,1);

    PolynomialFunction* bf = new PolynomialFunction(0, 1);
    DenseBasisVector basis;
    basis.push_back(bf);
    LinearApproximator regressor(basis.size(), basis);

    LQRsolver solver(lqr, regressor);

    solver.solve();
}
