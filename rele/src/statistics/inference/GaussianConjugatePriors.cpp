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

namespace ReLe
{

ParametricNormal GaussianConjugatePrior::compute(const arma::mat& Sigma,
        const ParametricNormal& prior,
        const arma::mat& samples)
{
    unsigned int n = samples.n_cols;
    arma::mat Sigma0 = prior.getCovariance();
    arma::vec mu0 = prior.getMean();

    arma::mat Sigma0inv = Sigma0.i();
    arma::mat Sigmainv = Sigma.i();

    arma::vec muSample = arma::mean(samples, 1);

    arma::mat SigmaPostinv = Sigma0inv + n*Sigmainv;
    arma::vec muPost = arma::solve(SigmaPostinv, Sigma0inv*mu0 + n*Sigmainv*muSample);
    arma::mat SigmaPost = SigmaPostinv.i();

    return ParametricNormal(muPost, SigmaPost);
}

Wishart GaussianConjugatePrior::compute(const arma::vec& mean,
                                        const Wishart& prior,
                                        const arma::mat& samples)
{
    unsigned int n = samples.n_cols;

    unsigned int nuPost = n + prior.getNu();

    arma::mat Xn = samples.each_col() - arma::mean(samples, 1);

    arma::mat VPost = (prior.getV().i() + Xn*Xn.t()).i();

    return Wishart(nuPost, VPost);
}

InverseWishart GaussianConjugatePrior::compute(const arma::vec& mean,
        const InverseWishart& prior,
        const arma::mat& samples)
{
    unsigned int n = samples.n_cols;

    unsigned int nuPost = n + prior.getNu();

    arma::mat Xn = samples.each_col() - arma::mean(samples, 1);

    arma::mat VPost = (prior.getPsi() + Xn*Xn.t());

    return InverseWishart(nuPost, VPost);
}


}
