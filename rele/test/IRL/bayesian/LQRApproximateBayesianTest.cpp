/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/tiles/BasicTiles.h"

#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/LQR.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/utils/FileManager.h"

#include "rele/IRL/algorithms/BayesianCoordinateAscend.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    int nbEpisodes = 100;

    FileManager fm("lqr", "approximateBayesian");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    //Set reward policy
    vec eReward = {0.2, 0.7, 0.1};

    // Build policy
    int rewardDim = eReward.n_elem;
    int dim = 2;
    LQR mdp(dim, rewardDim);

    BasisFunctions basis = IdentityBasis::generate(dim);

    SparseFeatures phi;
    phi.setDiagonal(basis);

    DetLinearPolicy<DenseState> expertPolicy(phi);

    // solve the problem in exact way
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    arma::mat Sigma = arma::eye(dim, dim);
    Sigma *= 0.001;
    ParametricNormal expertDist(p, Sigma);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertDist.getParameters().t() << std::endl;


    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    // Generate LQR expert dataset
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;

    // recover approximate
    std::vector<Range> ranges;
    std::vector<unsigned int> tilesN;
    unsigned int numTiles = 3;

    for(unsigned int i = 0; i < dim; i++)
    {
        ranges.push_back(Range(-12, 12));
        tilesN.push_back(numTiles);
    }

    BasicTiles* tiles = new BasicTiles(ranges, tilesN);

    DenseTilesCoder phiImitator(tiles, dim);

    unsigned int dp = phiImitator.rows();

    std::cout << dp << std::endl;

    // mean prior
    arma::vec mu_p(dp, arma::fill::zeros);
    arma::mat Sigma_p = arma::eye(dp, dp)*10;
    ParametricNormal prior(mu_p, Sigma_p);

    // Covariance prior (fixed)
    arma::mat SigmaImitator = arma::eye(dp, dp);
    SigmaImitator *= 0.1;

    arma::mat SigmaPolicy = arma::eye(dim, dim);
    MVNPolicy policyFamily(phiImitator, SigmaPolicy);
    BayesianCoordinateAscendMean<DenseAction, DenseState> alg(policyFamily, prior, SigmaImitator);

    std::cout << "Recovering Distribution" << std::endl;
    alg.compute(data);

    ParametricNormal posterior = alg.getPosterior();

    std::cout << "Mean parameters" << std::endl
              << posterior.getMean().t() << std::endl
              << "Covariance estimate" << std::endl
              << posterior.getCovariance() << std::endl;

}
