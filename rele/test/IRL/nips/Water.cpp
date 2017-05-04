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

#include "rele/approximators/features/SparseFeatures.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/TilesCoder.h"
#include "rele/approximators/regressors/others/GaussianMixtureModels.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/tiles/BasicTiles.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/WaterResources.h"

#include "rele/core/PolicyEvalAgent.h"
#include "rele/core/Core.h"

#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/SDPEGIRL.h"
#include "rele/IRL/algorithms/CurvatureEGIRL.h"
#include "rele/IRL/algorithms/SCIRL.h"
#include "rele/IRL/algorithms/CSI.h"

#include "rele/utils/FileManager.h"
#include "rele/utils/DatasetDiscretizator.h"
#include "rele/core/callbacks/CoreCallback.h"

using namespace std;
using namespace arma;
using namespace ReLe;

class WaterRewardFeatures: public Features
{
public:
    WaterRewardFeatures(const WaterResourcesSettings* config)
        : config(config)
    {

    }

    virtual arma::mat operator()(const arma::vec& input) const override
    {
        arma::vec x = input.rows(0, 1);
        arma::vec xn = input.rows(4, 5);

        arma::vec r = input.rows(2, 3);
        arma::vec reward(4);

        arma::vec h = xn / config->S;
        double P = computePowerGeneration(h[0], r[0]);

        reward[0] = -std::pow(std::max(h[0] - config->h_flo[0], 0.0), 2);
        reward[1] = -std::pow(std::max(h[1] - config->h_flo[1], 0.0), 2);
        reward[2] = -std::pow(std::max(config->w_irr - r[1], 0.0), 2);
        reward[3] = -std::max(config->w_hyd - P, 0.0);

        return reward;
    }

    virtual size_t rows() const override
    {
        return 4;
    }

    virtual size_t cols() const override
    {
        return 1;
    }

private:
    double computePowerGeneration(double h, double r) const
    {
        const double g = 9.81;
        double q = std::max(r - config->q_mef, 0.0);

        return config->eta*g*config->gamma_h20*h * q / (3.6e6);
    }

    const WaterResourcesSettings* config;
};

int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        cout << "Wrong argument number: n_episode, n_discretizations, n_experiment must be provided" << endl;
        return -1;
    }

    string n_episodes(argv[1]);
    string n_discretizations(argv[2]);
    string n_experiment(argv[3]);

    FileManager fm("nips/water/"+n_episodes+"/"+n_experiment);
    fm.cleanDir();
    fm.createDir();
    std::cout << std::setprecision(OS_PRECISION);

    unsigned int nbEpisodes = std::stoi(n_episodes);


    WaterResources mdp;
    std::vector<Range> ranges;
    ranges.push_back(Range(0, 101));
    ranges.push_back(Range(0, 101));

    std::vector<unsigned int> tilesN = {10, 10};

    BasicTiles* tiles = new BasicTiles(ranges, tilesN);

    DenseTilesCoder phi(tiles, 2);

    arma::vec p;
    p.load("/tmp/ReLe/nips/water/Weights.txt", arma::raw_ascii);

    DetLinearPolicy<DenseState> expertPolicy(phi);
    ParametricNormal expertDist(p, arma::eye(p.size(), p.size()));

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    // -- Generate expert dataset -- //
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    arma::mat theta = expert.getParams();

    // -- RUN -- //

    // Create parametric reward
    WaterRewardFeatures phiReward(static_cast<const WaterResourcesSettings*>(&mdp.getSettings()));

    LinearApproximator rewardRegressor(phiReward);


    unsigned int numAlg = 3;
    IRLAlgorithm<DenseAction, DenseState>* irlAlg[numAlg];


    irlAlg[0] = new EGIRL<DenseAction,DenseState>(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);
    irlAlg[1] = new SDPEGIRL<DenseAction,DenseState>(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);
    irlAlg[2] = new CurvatureEGIRL<DenseAction,DenseState>(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE, 1e-4);

    arma::mat weights(rewardRegressor.getParametersSize(), numAlg + 2, arma::fill::zeros);

    for(unsigned int i = 0; i < numAlg; i++)
    {
        irlAlg[i]->run();
        weights.col(i) = rewardRegressor.getParameters();
        delete irlAlg[i];
    }

    //Classification Based
    unsigned int discretizedActions = std::stoi(n_discretizations);

    IRLAlgorithm<FiniteAction, DenseState>* irlAlg_c[2];
    std::vector<Range> ranges_c;
    ranges_c.push_back(Range(0, 101));
    ranges_c.push_back(Range(0, 101));
    ranges_c.push_back(Range(0, 101));
    ranges_c.push_back(Range(0, 101));

    std::vector<unsigned int> tilesN_c = {10, 10, 10, 10};
    std::vector<unsigned int> stateComponents = {0, 1, 4, 5};
    BasicTiles* tiles_c = new SelectiveTiles(stateComponents, ranges_c, tilesN_c);
    DenseTilesCoder phi_c(tiles_c);

    std::vector<unsigned int> actionsPerDim = {14, 14};
    std::vector<Range> ranges_d;
    ranges_d.push_back(Range(-30, 140));
    ranges_d.push_back(Range(-30, 140));
    DatasetDiscretizator discretizator(ranges_d, actionsPerDim);
    auto&& discretizedData = discretizator.discretize(data);

    /*irlAlg_c[0] = new SCIRL<DenseState>(discretizedData,rewardRegressor, mdp.getSettings().gamma, discretizedActions);*/
    irlAlg_c[0] = new CSI<DenseState>(discretizedData, phi_c, rewardRegressor, mdp.getSettings().gamma, discretizedActions);

    for(unsigned int i = 0; i < 1; i++)
    {
        irlAlg_c[i]->run();
        weights.col(numAlg+i) = rewardRegressor.getParameters();
        delete irlAlg_c[i];
    }

    // -- Save weights -- //
    weights.save(fm.addPath("RewardWeights.txt"), arma::raw_ascii);

    return 0;
}
