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

#include "rele/policy/parametric/differentiable/PortfolioNormalPolicy.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/environments/Portfolio.h"
#include "PGTest.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    CommandLineParser clp;
    gradConfig config = clp.getConfig(argc, argv);
    config.envName = "portfolio";

    Portfolio mdp;

    int dim = mdp.getSettings().stateDimensionality;

    //--- define policy (low level)
    BasisFunctions basis = IdentityBasis::generate(dim);

    DenseFeatures phi(basis);

    double epsilon = 0.05;
    PortfolioNormalPolicy policy(epsilon, phi);
    //---

    PGTest<FiniteAction, DenseState> pgTest(config, mdp, policy);
    pgTest.run();

    return 0;
}
