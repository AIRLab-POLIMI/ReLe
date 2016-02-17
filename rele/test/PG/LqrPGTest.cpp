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

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/BasisFunctions.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"

#include "rele/environments/LQR.h"
#include "PGTest.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    CommandLineParser clp;
    gradConfig config = clp.getConfig(argc, argv);
    config.envName = "lqr";

    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)

    IdentityBasis* pf = new IdentityBasis(0);
    cout << *pf << endl;
    DenseFeatures phi(pf);
    NormalPolicy policy(0.1, phi);

    PGTest<DenseAction, DenseState> pgTest(config, mdp, policy);
    pgTest.run();

    delete pf;

    return 0;
}
