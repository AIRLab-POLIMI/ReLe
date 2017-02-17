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


#include "rele/algorithms/policy_search/PGPE/PGPE.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/approximators/tiles/BasicTiles.h"
#include "rele/approximators/features/TilesCoder.h"
#include "rele/utils/FileManager.h"

#include "rele/environments/WaterResources.h"

using namespace ReLe;

int main(int argc, char** argv)
{
    FileManager fm("WaterResources", "PGPE");
    fm.createDir();


    //Build environment
    WaterResources mdp;

    //Build policy
    std::vector<Range> ranges;
    ranges.push_back(Range(0, 101));
    ranges.push_back(Range(0, 101));

    std::vector<unsigned int> tilesN = {4, 4};

    BasicTiles* tiles = new BasicTiles(ranges, tilesN);

    DenseTilesCoder phi(tiles, 2);
    DetLinearPolicy<DenseState> policy(phi);

    //Build distribution
    arma::vec mean = arma::ones(phi.rows())*50;
    arma::mat Sigma = arma::eye(phi.rows(), phi.rows())*0.05;
    ParametricNormal dist(mean, Sigma);

    //Build Reward Transformation
    //arma::vec rewardWeights = {0.5, 0.5, 0.0, 0.0};
    arma::vec rewardWeights = arma::ones(4)/4.0;
    WeightedSumRT rewardT(rewardWeights);

    //Build agent
    AdaptiveGradientStep step(0.1);
    //ConstantGradientStep step(0.1);
    PGPE<DenseAction, DenseState> agent(dist, policy, 1, 10, step, rewardT);

    //Run experiments
    Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().episodeN = 10000000;
    core.getSettings().testEpisodeN = 100;
    core.getSettings().episodeLength = mdp.getSettings().horizon;

    arma::vec J = core.runEvaluation();
    std::cout << "J: " << J.t() << " " << J.t()*rewardWeights << std::endl;

    core.runEpisodes();

    J = core.runEvaluation();
    std::cout << "J: " << J.t() << " " << J.t()*rewardWeights << std::endl;

    WriteStrategy<DenseAction, DenseState> wStrategy(fm.addPath("dataset.log"),
            WriteStrategy<DenseAction, DenseState>::TRANS, true);
    core.getSettings().loggerStrategy = &wStrategy;

    core.runTestEpisodes();

    return 0;
}
