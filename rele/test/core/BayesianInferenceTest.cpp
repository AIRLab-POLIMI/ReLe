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
    arma::vec mu = {0.5, 2.3};
    arma::mat Sigma = {{0.2, 0.1},{0.1, 0.2}};

    arma::mat samples = mvnrand(samplesN, mu, Sigma);

    arma::vec samplesMean = arma::mean(samples, 1);
    arma::mat samplesCov = arma::cov(samples.t());

    arma::vec mu_p(2, arma::fill::zeros);
    arma::mat Sigma_p = arma::eye(2, 2)*10;
    ParametricNormal prior(mu_p, Sigma_p);

    arma::mat V_p = arma::eye(2, 2)*1e5;
    arma::mat Psi_p = V_p.i();

    Wishart precPrior(2, V_p);
    InverseWishart covPrior(2, Psi_p);

    std::cout << "GT" << std::endl;
    std::cout << mu.t() << std::endl;
    std::cout << Sigma << std::endl;

    std::cout << "Maximum Likelihood" << std::endl;
    std::cout << samplesMean.t() << std::endl;
    std::cout << samplesCov << std::endl;


    std::cout << "Mean estimation test" << std::endl;
    ParametricNormal&& posterior = GaussianConjugatePrior::compute(Sigma, prior, samples);

    std::cout << "prior" << std::endl;
    std::cout << prior.getMean().t() << std::endl;
    std::cout << prior.getCovariance() << std::endl;

    std::cout << "posterior" << std::endl;
    std::cout << posterior.getMean().t() << std::endl;
    std::cout << posterior.getCovariance() << std::endl;


    std::cout << "Precision estimation test" << std::endl;
    Wishart&& precPosterior = GaussianConjugatePrior::compute(mu, precPrior, samples);

    std::cout << "prior" << std::endl;
    std::cout << precPrior.getV() << std::endl;
    std::cout << precPrior.getNu() << std::endl;

    std::cout << "posterior" << std::endl;
    std::cout << precPosterior.getV() << std::endl;
    std::cout << precPosterior.getNu() << std::endl;

    std::cout << "posterior covariance mean" << std::endl;
    std::cout << precPosterior.getMean().i() << std::endl;

    std::cout << "Covariance estimation test" << std::endl;
    InverseWishart&& covPosterior = GaussianConjugatePrior::compute(mu, covPrior, samples);

    std::cout << "prior" << std::endl;
    std::cout << covPrior.getPsi() << std::endl;
    std::cout << covPrior.getNu() << std::endl;

    std::cout << "posterior" << std::endl;
    std::cout << covPosterior.getPsi() << std::endl;
    std::cout << covPosterior.getNu() << std::endl;

    std::cout << "posterior covariance mean" << std::endl;
    std::cout << covPosterior.getMean() << std::endl;
}
