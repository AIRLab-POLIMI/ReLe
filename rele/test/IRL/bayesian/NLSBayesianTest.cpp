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

#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/IdentityBasis.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/NLS.h"

#include "rele/core/PolicyEvalAgent.h"
#include "rele/core/Core.h"

#include "rele/utils/FileManager.h"

#include "rele/IRL/algorithms/BayesianCoordinateAscend.h"

using namespace std;
using namespace arma;
using namespace ReLe;

//#define PRINT
#define RUN_GIRL
#define RECOVER

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    unsigned int nbEpisodes = 100;

    FileManager fm("nls", "bayesian");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    NLS mdp;

    //Setup expert policy
    unsigned int dim = mdp.getSettings().stateDimensionality;
    unsigned int actionDim = 1;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    arma::vec p(2);
    p(0) = 6.5178;
    p(1) = -2.5994;

    DetLinearPolicy<DenseState> expertPolicy(phi);
    arma::mat Sigma = arma::eye(dim, dim)*0.1;
    ParametricNormal expertDist(p, Sigma);

    std::cout << "Mean gt: " << expertDist.getParameters().t() << std::endl;
    std::cout << "Sigma gt: " << std::endl
              << Sigma;

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    // Generate expert dataset
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;

    // recover initial policy
    arma::mat SigmaPolicy = arma::eye(actionDim, actionDim)*0.01;
    MVNPolicy policyFamily(phi, SigmaPolicy);

    // priors
    arma::vec mu_p = {0.0, 0.0};
    arma::mat Sigma_p = arma::eye(dim, dim)*10;
    ParametricNormal meanPrior(mu_p, Sigma_p);

    arma::mat Psi = arma::eye(2, 2);
    unsigned int nu = 2;
    InverseWishart covPrior(nu, Psi);

    std::cout << "initial covariance mode" << std::endl;
    std::cout << covPrior.getMode() << std::endl;


    BayesianCoordinateAscendMean<DenseAction, DenseState> alg(policyFamily, meanPrior, Sigma);

    std::cout << "Recovering Distribution (mean only)" << std::endl;
    alg.compute(data);

    ParametricNormal posterior = alg.getPosterior();

    std::cout << "Mean parameters" << std::endl
              << posterior.getMean().t() << std::endl
              << "Covariance estimate" << std::endl
              << posterior.getCovariance() << std::endl;



    BayesianCoordinateAscendFull<DenseAction, DenseState> alg2(policyFamily, meanPrior, covPrior);

    std::cout << "Recovering Distribution (mean and covariance)" << std::endl;
    alg2.compute(data);

    ParametricNormal meanPosterior = alg2.getMeanPosterior();
    InverseWishart covPosterior = alg2.getCovPosterior();

    std::cout << "Mean" << std::endl
              << meanPosterior.getMean().t() << std::endl
              << "Covariance estimate" << std::endl
              << meanPosterior.getCovariance() << std::endl;

    std::cout << "Cov parameters" << std::endl
              << covPosterior.getMode() << std::endl;

}
