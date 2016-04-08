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

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/environments/ShipSteering.h"
#include "PGTest.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    CommandLineParser clp;
    gradConfig config = clp.getConfig(argc, argv);
    config.envName = "ship";

    ShipSteering mdp;

    int dim = mdp.getSettings().stateDimensionality;

    //--- define policy (low level)
    BasisFunctions basis = GaussianRbf::generate(
    {
        3,
        3,
        6,
        2
    },
    {
        0.0, 150.0,
        0.0, 150.0,
        -M_PI, M_PI,
        -15.0, 15.0
    });

    DenseFeatures phi(basis);

    double epsilon = 0.05;
    NormalPolicy policy(epsilon, phi);
    //---

    PGTest<DenseAction, DenseState> pgTest(config, mdp, policy);
    pgTest.run();

    /*delete core.getSettings().loggerStrategy;

    //--- collect some trajectories
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath(outputname),  WriteStrategy<DenseAction, DenseState>::TRANS,false);
    core.getSettings().testEpisodeN = 3000;
    core.runTestEpisodes();
    //---

    delete core.getSettings().loggerStrategy;*/

    return 0;
}
