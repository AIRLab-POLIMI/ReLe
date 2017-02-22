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
#include "rele/algorithms/policy_search/em/episode_based/RWR.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/approximators/tiles/BasicTiles.h"
#include "rele/approximators/features/TilesCoder.h"
/*#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/features/DenseFeatures.h"*/
#include "rele/utils/FileManager.h"

#include "rele/environments/WaterResources.h"

using namespace ReLe;

#define LOAD

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

    std::vector<unsigned int> tilesN = {10, 10};

    BasicTiles* tiles = new BasicTiles(ranges, tilesN);

    DenseTilesCoder phi(tiles, 2);
    /*auto basis = PolynomialFunction::generate(2, 2);
    DenseFeatures phi(basis);*/


    DetLinearPolicy<DenseState> policy(phi);

    //Build distribution
    arma::vec mean;
#ifdef LOAD
    mean.load(fm.addPath("Weights.txt"));
#else
    arma::vec mean = arma::ones(phi.rows())*50;
#endif
    //arma::vec mean = arma::zeros(phi.rows());
    arma::mat Sigma = arma::eye(phi.rows(), phi.rows())*5;
    arma::mat SigmaEv = arma::eye(phi.rows(), phi.rows())*0.01;
    ParametricNormal dist(mean, Sigma);

    //ParametricCholeskyNormal dist(mean, arma::chol(Sigma));

    //Build Reward Transformation
    //arma::vec rewardWeights = {0.5, 0.5, 0.0, 0.0};
    arma::vec rewardWeights = arma::ones(4)/4.0;
    WeightedSumRT rewardT(rewardWeights);

    //Build agent
    AdaptiveGradientStep step(0.1);
    //ConstantGradientStep step(0.1);
    PGPE<DenseAction, DenseState> agent(dist, policy, 1, 10, step, rewardT);

    //Run experiments
    unsigned int nbUpdates = 10000;
    unsigned int nbepperpol = 1;
    unsigned int nbpolperupd = 100;

    Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().testEpisodeN = 100;
    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().testEpisodeN = 100;

    arma::vec J = core.runEvaluation();
    std::cout << "J: " << J.t() << " " << J.t()*rewardWeights << std::endl;

    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    double every, bevery;
    every = bevery = 0.1; //%
    int updateCount = 0;
    for (int i = 0; i < episodes; i++)
    {
    	dist.setCovariance(Sigma);
        core.runEpisode();

        int v = nbepperpol*nbpolperupd;
        if (i % v == 0)
        {
            updateCount++;
            if ((updateCount >= nbUpdates*every) || (updateCount == 1))
            {
            	dist.setCovariance(SigmaEv);
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
                std::cout << "### " << p << "% ###" << std::endl;
                arma::vec J = core.runEvaluation();
                std::cout << "mean score: " << J.t()*rewardWeights << std::endl;
                std::cout << "objective score:" << J.t() << std::endl;
                every += bevery;
            }
        }
    }

    dist.setCovariance(SigmaEv);
    dist.getParameters().save(fm.addPath("Weights.txt"),  arma::raw_ascii);

    WriteStrategy<DenseAction, DenseState> strategy(fm.addPath("Trajectories.txt"),
    			WriteStrategy<DenseAction, DenseState>::TRANS, true);
    core.getSettings().loggerStrategy = &strategy;

    core.runTestEpisodes();

    return 0;
}
