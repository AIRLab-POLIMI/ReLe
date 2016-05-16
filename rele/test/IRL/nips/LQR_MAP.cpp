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
#include "rele/approximators/basis/PolynomialFunction.h"
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
#include "rele/IRL/algorithms/LinearMLEDistribution.h"
#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/SDPEGIRL.h"
#include "rele/IRL/algorithms/CurvatureEGIRL.h"

#include "../RewardBasisLQR.h"

#include "rele/algorithms/policy_search/NES/NES.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    int nbEpisodes = 100;

    FileManager fm("nips", "lqr_mle");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    //Build MDP
    vec eReward = {0.3, 0.7};

    int rewardDim = eReward.n_elem;
    int dim = 2;
    LQR mdp(dim, rewardDim, LQR::S0Type::RANDOM);


    // Build policy
    std::vector<Range> ranges;
    std::vector<unsigned int> tilesN;
    unsigned int numTiles = 5;

    for(unsigned int i = 0; i < dim; i++)
    {
        ranges.push_back(Range(-3, 3));
        tilesN.push_back(numTiles);
    }

    Tiles* tiles = new CenteredLogTiles(ranges, tilesN);
    DenseTilesCoder phi(tiles, dim);

    DetLinearPolicy<DenseState> expertPolicy(phi);

    // solve the problem
    unsigned int dp = expertPolicy.getParametersSize();

    arma::mat SigmaExpert = arma::eye(dp, dp)*1e-2;

    arma::vec muExpert;
    muExpert.load("/tmp/ReLe/nips/lqr_map/ExpertWeights.dat");
    ParametricNormal expertDist(muExpert, SigmaExpert);

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    /*arma::vec p0 = arma::zeros(dp);
    ParametricNormal expertDist(p0, SigmaExpert);
    AdaptiveGradientStep stepLenght(0.01);
    WeightedSumRT rewardTransformation(eReward);

    NES<DenseAction, DenseState> expert(expertDist, expertPolicy, 1, 10, stepLenght, rewardTransformation);*/

    // Generate LQR expert dataset
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().episodeN = 10000;
    expertCore.getSettings().testEpisodeN = nbEpisodes;

    /*expertCore.runEpisodes();
    std::cout << "J: " << std::endl;
    std::cout << eReward.t()*expertCore.runEvaluation() << std::endl;*/

    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    //-- ################################################################### --//

    // mean prior
    arma::vec mu_p = arma::zeros(dp);
    arma::mat Sigma_p = arma::eye(dp, dp)*1e1;
    ParametricNormal prior(mu_p, Sigma_p);

    // Covariance prior
    unsigned int nu = dp+2;
    arma::mat V = arma::eye(dp, dp)*1e1;
    Wishart covPrior(nu, V);

    //Policy
    arma::mat SigmaPolicy = arma::eye(dim, dim)*1e-3;
    MVNPolicy policyFamily(phi, SigmaPolicy);

    // Recover policy
    LinearMLEDistribution algMLE(phi, SigmaPolicy);
    LinearBayesianCoordinateAscendFull algMAP(policyFamily, phi, SigmaPolicy, prior, covPrior);

    algMLE.compute(data);
    algMAP.compute(data);

    auto distMLE = algMLE.getDistribution();
    auto distMAP = algMAP.getDistribution();

    std::cout << "GT" << std::endl
              << " mean: "  << std::endl
              << expertDist.getMean().t()
              << " cov: " << std::endl
              << expertDist.getCovariance().diag().t() << std::endl;

    std::cout << "MLE" << std::endl
              << " mean: "  << std::endl
              << distMLE.getMean().t()
              << " cov: " << std::endl
              << distMLE.getCovariance().diag().t() << std::endl;

    std::cout << "MAP" << std::endl
              << " mean: "  << std::endl
              << distMAP.getMean().t()
              << " cov: " << std::endl
              << distMAP.getCovariance().diag().t() << std::endl;

    std::cout << "error MLE - mean: "
              << arma::norm(expertDist.getMean()-distMLE.getMean())
              << " cov: "
              << arma::norm(expertDist.getCovariance()-distMLE.getCovariance()) << std::endl;

    std::cout << "error MAP - mean: "
              << arma::norm(expertDist.getMean()-distMAP.getMean())
              << " cov: "
              << arma::norm(expertDist.getCovariance()-distMAP.getCovariance()) << std::endl;


    //-- ################################################################### --//

    arma::mat thetaMLE = algMLE.getParameters();
    arma::mat thetaMAP = algMAP.getParameters();

    BasisFunctions basisReward;
    for(unsigned int i = 0; i < eReward.n_elem; i++)
        basisReward.push_back(new LQR_RewardBasis(i, dim));
    DenseFeatures phiReward(basisReward);
    LinearApproximator rewardRegressor(phiReward);

    auto* irlAlgMLE =  new EGIRL<DenseAction, DenseState>(data, thetaMLE, distMLE,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);

    auto* irlAlgMAP =  new EGIRL<DenseAction, DenseState>(data, thetaMAP, distMAP,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);

    //Run GIRL
    irlAlgMLE->run();
    arma::vec omegaMLE = rewardRegressor.getParameters();

    irlAlgMAP->run();
    arma::vec omegaMAP = rewardRegressor.getParameters();

    //Print results
    cout << "Weights (MLE): " << omegaMLE.t();
    cout << "Weights (MAP): " << omegaMAP.t();


}
