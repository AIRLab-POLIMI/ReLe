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
#include "rele/approximators/tiles/LogTiles.h"

#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/LQR.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/utils/FileManager.h"

#include "rele/IRL/algorithms/BayesianCoordinateAscend.h"
#include "rele/IRL/algorithms/MLEDistribution.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    int nbEpisodes = 300;

    FileManager fm("lqr", "approximateBayesian");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    //Set reward policy
    vec eReward = {0.2, 0.7, 0.1};

    // Build policy
    int rewardDim = eReward.n_elem;
    int dim = 2;
    LQR mdp(dim, rewardDim, LQR::S0Type::RANDOM);

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
    Sigma *= 1e-3;
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
    unsigned int numTiles = 4;

    for(unsigned int i = 0; i < dim; i++)
    {
        ranges.push_back(Range(-4, 4));
        tilesN.push_back(numTiles);
    }

    Tiles* tiles = new BasicTiles(ranges, tilesN);
    //Tiles* tiles = new CenteredLogTiles(ranges, tilesN);

    DenseTilesCoder phiImitator(tiles, dim);

    unsigned int dp = phiImitator.rows();

    std::cout << dp << std::endl;

    // mean prior
    arma::vec mu_p = arma::zeros(dp);
    arma::mat Sigma_p = arma::eye(dp, dp)*1e2;
    ParametricNormal prior(mu_p, Sigma_p);

    // Covariance prior (fixed)
    arma::mat Psi = arma::eye(dp, dp)*1e3;
    unsigned int nu = dp+1;
    InverseWishart covPrior(nu, Psi);

    arma::mat SigmaPolicy = arma::eye(dim, dim)*1e-3;
    MVNPolicy policyFamily(phiImitator, SigmaPolicy);
    //BayesianCoordinateAscendFull<DenseAction, DenseState> alg(policyFamily, prior, covPrior);

    MLEDistribution<DenseAction, DenseState> alg(policyFamily);

    std::cout << "Recovering Distribution" << std::endl;
    alg.compute(data);



    ParametricNormal imitatorDist = alg.getDistribution();

    std::cout << "Mean parameters" << std::endl
              << imitatorDist.getMean().t() << std::endl;
    /*<< "Covariance estimate" << std::endl
    << posterior.getCovariance() << std::endl;*/

    // Generate LQR imitator dataset
    DetLinearPolicy<DenseState> detPolicyFamily(phiImitator);
    PolicyEvalDistribution<DenseAction, DenseState> imitator(imitatorDist, detPolicyFamily);
    Core<DenseAction, DenseState> imitatorCore(mdp, imitator);
    CollectorStrategy<DenseAction, DenseState> collectionImitator;
    imitatorCore.getSettings().loggerStrategy = &collectionImitator;
    imitatorCore.getSettings().episodeLength = mdp.getSettings().horizon;
    imitatorCore.getSettings().testEpisodeN = nbEpisodes;
    imitatorCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& imitatorData = collectionImitator.data;

    //Save trajectories
    std::ofstream ofs(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs);
    std::ofstream ofs2(fm.addPath("TrajectoriesImitator.txt"));
    imitatorData.writeToStream(ofs2);


}
