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

#include "rele/feature_selection/PrincipalComponentAnalysis.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        cout << "Wrong argument number: dimension, n_episode, n_experiment must be provided" << endl;
        return -1;
    }

    string dimension(argv[1]);
    string n_episodes(argv[2]);
    string n_experiment(argv[3]);

    FileManager fm("nips/lqr_map2/" + dimension + "/" + n_episodes + "/" + n_experiment);
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    unsigned int dim = stoi(dimension);
    unsigned int nbEpisodes = stoi(n_episodes);

    //Set reward policy
    arma::vec eReward;

    if(dim == 2)
    {
        eReward = {0.3, 0.7};
    }
    else
    {
        eReward = {0.2, 0.7, 0.1};
    }

    int rewardDim = dim;
    LQR mdp(dim, rewardDim, LQR::S0Type::RANDOM);

    BasisFunctions basis = IdentityBasis::generate(dim);

    SparseFeatures phi;
    phi.setDiagonal(basis);

    arma::mat Sigma = arma::eye(dim, dim)*1e-2;
    MVNPolicy expertPolicy(phi, Sigma);

    // solve the problem in exact way
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    arma::mat SigmaExpert = arma::eye(dim, dim)*1e-2;
    ParametricNormal expertDist(p, SigmaExpert);

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
    //BasisFunctions basisImitator = PolynomialFunction::generate(deg, dim);
    //SparseFeatures phiImitator(basisImitator, dim);
    BasisFunctions basisImitator = IdentityBasis::generate(dim);
    SparseFeatures phiImitator;
    phiImitator.setDiagonal(basisImitator);

    unsigned int dp = phiImitator.rows();

    std::cout << "Parameters Number" << std::endl;
    std::cout << dp << std::endl;

    //arma::mat SigmaPolicy = arma::eye(dim, dim);
    MVNPolicy policyFamily(phiImitator, SigmaExpert);
    LinearMLEDistribution algMLE(phiImitator, SigmaExpert);

    // mean prior
    arma::vec mu_p = arma::zeros(dp);
    arma::mat Sigma_p = arma::eye(dp, dp);
    ParametricNormal prior(mu_p, Sigma_p);


    // Covariance prior
    unsigned int nu = dp+2;
    arma::mat V = arma::eye(dp, dp)*1e3;
    Wishart covPrior(nu, V);
    LinearBayesianCoordinateAscendFull algMAP(policyFamily, phiImitator, SigmaExpert, prior, covPrior);

    std::cout << "Recovering Distribution (MLE)" << std::endl;
    algMLE.compute(data);

    std::cout << "Recovering Distribution (MAP)" << std::endl;
    algMAP.compute(data);

    ParametricNormal imitatorDistMLE = algMLE.getDistribution();
    ParametricNormal imitatorDistMAP = algMAP.getDistribution();

    arma::mat thetaExpert = expert.getParams();
    arma::mat thetaMLE = algMLE.getParameters();
    arma::mat thetaMAP = algMAP.getParameters();
    arma::mat meanMLE = imitatorDistMLE.getMean();
    arma::mat meanMAP = imitatorDistMAP.getMean();
    arma::mat covMLE = imitatorDistMLE.getCovariance();
    arma::mat covMAP = imitatorDistMAP.getCovariance();

    //save weights
    thetaExpert.save(fm.addPath("thetaExpert.txt"), arma::raw_ascii);
    thetaMLE.save(fm.addPath("thetaMLE.txt"), arma::raw_ascii);
    thetaMAP.save(fm.addPath("thetaMAP.txt"), arma::raw_ascii);
    meanMLE.save(fm.addPath("meanMLE.txt"), arma::raw_ascii);
    meanMAP.save(fm.addPath("meanMAP.txt"), arma::raw_ascii);
    covMLE.save(fm.addPath("covMLE.txt"), arma::raw_ascii);
    covMAP.save(fm.addPath("covMAP.txt"), arma::raw_ascii);

}
