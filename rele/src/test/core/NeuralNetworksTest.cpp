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
#include "regressors/FFNeuralNetwork.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    cout << "## Neural Network Test ##" << endl;

    arma::vec input = {1.0, 1.0};

    BasisFunctions basis = IdentityBasis::generate(input.size());
    DenseFeatures phi(basis);

    FFNeuralNetwork net(phi, 3, 1);
    arma::vec p = net.getParameters();

    arma::vec out1 = net(input);
    cout << out1 << endl;

    p.fill(1.0);
    net.setParameters(p);
    arma::vec out2 = net(input);
    cout << out2 << endl;

}
