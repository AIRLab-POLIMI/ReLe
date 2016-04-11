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

#include "rele/statistics/inference/GaussianConjugatePrior.h"
#include "rele/utils/ArmadilloPDFs.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    unsigned int samplesN = 10000;
    arma::vec mu_gt = {0.5, 2.3};
    arma::mat Sigma_gt = {{0.2, 0.1},{0.1, 0.2}};

    arma::mat samples = mvnrand(samplesN, mu_gt, Sigma_gt);

    arma::vec samplesMean = arma::mean(samples, 1);

    arma::mat Sigma = {{0.2, 0.1},{0.1, 0.2}};

    arma::vec mu_p(2, arma::fill::zeros);
    arma::mat Sigma_p(2, 2, arma::fill::eye);
    Sigma_p *= 10;
    ParametricNormal prior(mu_p, Sigma_p);

    ParametricNormal&& posterior = GaussianConjugatePrior::compute(Sigma, prior, samples);

    std::cout << "prior" << std::endl;
    std::cout << prior.getMean().t() << std::endl;
    std::cout << prior.getCovariance() << std::endl;

    std::cout << "posterior" << std::endl;
    std::cout << posterior.getMean().t() << std::endl;
    std::cout << posterior.getCovariance() << std::endl;

    std::cout << "Maximum Likelihood" << std::endl;
    std::cout << samplesMean.t() << std::endl;

    std::cout << "GT" << std::endl;
    std::cout << mu_gt.t() << std::endl;
    std::cout << Sigma_gt << std::endl;


}
