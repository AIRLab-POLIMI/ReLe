/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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
#include "rele/approximators/features/TilesCoder.h"
#include "rele/approximators/tiles/BasicTiles.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/environments/WaterResources.h"
#include "PGTest.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    CommandLineParser clp;
    gradConfig config = clp.getConfig(argc, argv);
    config.envName = "ship";


    //Build environment
    WaterResources mdp;

    //Build policy
    std::vector<Range> ranges;
    ranges.push_back(Range(0, 101));
    ranges.push_back(Range(0, 101));

    std::vector<unsigned int> tilesN = {10, 10};

    BasicTiles* tiles = new BasicTiles(ranges, tilesN);

    DenseTilesCoder phi(tiles, 2);

    int dim = mdp.getSettings().stateDimensionality;
    arma::vec mean = arma::ones(phi.rows())*50;
    //arma::vec mean = arma::zeros(phi.rows());
    arma::mat Sigma = arma::eye(2, 2)*10;

    MVNPolicy policy(phi, Sigma);
    policy.setParameters(mean);
    //---

    PGTest<DenseAction, DenseState> pgTest(config, mdp, policy);
    pgTest.run();

    return 0;
}
