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
#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/EMIRL.h"

#include "../RewardBasisLQR.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    int nbEpisodes = 1000;

    if(argc != 4)
    {
        cout << "wrong parameter numbers. must provide prior variance, "
             << "polynomial degree and experiment number" << endl;
        return -1;
    }


    std::string priorVarianceS(argv[1]);
    std::string polyDegreeS(argv[2]);
    std::string testN(argv[3]);

    double priorVariance = stod(priorVarianceS);
    unsigned int polyDegree = stoi(polyDegreeS);

    FileManager fm("approximateBayesianTest/"+priorVarianceS+"/"+polyDegreeS+"/"+ testN);
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    //Set reward policy
    vec eReward = {0.3, 0.7};

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
    arma::mat SigmaExpert = arma::eye(dim, dim)*1e-3;
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
    BasisFunctions basisImitator = PolynomialFunction::generate(polyDegree, dim);
    //basisImitator.erase(basisImitator.begin());
    SparseFeatures phiImitator(basisImitator, dim);

    unsigned int dp = phiImitator.rows();

    std::cout << "Parameters Number" << std::endl;
    std::cout << dp << std::endl;
    std::cout << "Feature expectation" << std::endl;
    std::cout << data.computefeatureExpectation(phiImitator) << std::endl;

    // mean prior
    arma::vec mu_p = arma::zeros(dp);
    arma::mat Sigma_p = arma::eye(dp, dp)*priorVariance;
    ParametricNormal prior(mu_p, Sigma_p);

    // Covariance prior (fixed)
    arma::mat Sigma = arma::eye(dp, dp)*1e-4;

    // Covariance prior
    arma::mat Psi = arma::eye(dp, dp)*1e3;
    unsigned int nu = dp+1;
    InverseWishart covPrior(nu, Psi);

    arma::mat SigmaPolicy = arma::eye(dim, dim)*1e-3;
    MVNPolicy policyFamily(phiImitator, SigmaPolicy);
    BayesianCoordinateAscendMean<DenseAction, DenseState> alg(policyFamily, prior, Sigma);

    std::cout << "Recovering Distribution" << std::endl;
    alg.compute(data);

    ParametricNormal imitatorDist = alg.getDistribution();
    arma::vec meanParametrers = imitatorDist.getMean();

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

    //Recover reward weights
    arma::mat theta = alg.getParameters();

    /* Create parametric reward */
    BasisFunctions basisReward;
    for(unsigned int i = 0; i < eReward.n_elem; i++)
        basisReward.push_back(new LQR_RewardBasis(i, dim));
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);
    auto* irlAlg =  new EGIRL<DenseAction, DenseState>(data, theta, imitatorDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);



    //Run EGIRL
    irlAlg->run();
    arma::vec omega = rewardRegressor.getParameters();

    omega.save(fm.addPath("Weights.txt"), arma::raw_ascii);
    p.save(fm.addPath("Expert.txt"), arma::raw_ascii);
    imitatorDist.getMean().save(fm.addPath("Imitator.txt"), arma::raw_ascii);

}
