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
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/features/DenseFeatures.h"

#include "rele/environments/Dam.h"
#include "PGTest.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    CommandLineParser clp;
    gradConfig config = clp.getConfig(argc, argv);
    config.envName = "dam";

    Dam mdp;

    PolynomialFunction *pf = new PolynomialFunction();
    GaussianRbf* gf1 = new GaussianRbf(0, 50, true);
    GaussianRbf* gf2 = new GaussianRbf(50, 20, true);
    GaussianRbf* gf3 = new GaussianRbf(120, 40, true);
    GaussianRbf* gf4 = new GaussianRbf(160, 50, true);
    BasisFunctions basis;
    basis.push_back(pf);
    basis.push_back(gf1);
    basis.push_back(gf2);
    basis.push_back(gf3);
    basis.push_back(gf4);

    DenseFeatures phi(basis);
    MVNLogisticPolicy policy(phi, 50);
    vec p(6);
    p(0) = 50;
    p(1) = -50;
    p(2) = 0;
    p(3) = 0;
    p(4) = 50;
    p(5) = 0;
    policy.setParameters(p);

    PGTest<DenseAction, DenseState> pgTest(config, mdp, policy);
    pgTest.run();

    return 0;
}
