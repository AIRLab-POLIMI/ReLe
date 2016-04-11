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

#ifndef INCLUDE_RELE_STATISTICS_INFERENCE_GAUSSIANCONJUGATEPRIOR_H_
#define INCLUDE_RELE_STATISTICS_INFERENCE_GAUSSIANCONJUGATEPRIOR_H_

#include "rele/statistics/DifferentiableNormals.h"

namespace ReLe
{

/*!
 * This class contains some methods to compute the posterior of
 * a multivariate gaussian distribution:
 *
 * \f[
 * p(\omega|\mathcal(D))\sim p(\mathcal(D)|\omega)*p(\omega)
 * \f]
 *
 * Every function implements a different type of problem, depending on which parameters
 * are known (mean, variance, nothing).
 *
 */
class GaussianConjugatePrior
{
public:
    /*!
     * Computes the distribution posterior when the variance is known.
     */
    static ParametricNormal compute(const arma::mat& Sigma,
                                    const ParametricNormal& prior,
                                    const arma::mat& samples);

};


}


#endif /* INCLUDE_RELE_STATISTICS_INFERENCE_GAUSSIANCONJUGATEPRIOR_H_ */
