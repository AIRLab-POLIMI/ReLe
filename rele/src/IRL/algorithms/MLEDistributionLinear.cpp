/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#include "rele/IRL/algorithms/MLEDistributionLinear.h"

namespace ReLe
{

MLEDistributionLinear::MLEDistributionLinear(Features& phi)
    : phi(phi)
{

}

void MLEDistributionLinear::compute(const Dataset<DenseAction, DenseState>& data)
{
    unsigned int n = data.size();

    params.zeros(phi.rows()*data[0][0].u.n_elem, n);

    //Compute policy MLE for each element
    for (unsigned int ep = 0; ep < data.size(); ep++)
    {
        Dataset<DenseAction, DenseState> epDataset;
        epDataset.push_back(data[ep]);
        LinearStatisticEstimation mleCalculator;
        try
        {
            mleCalculator.computeMLE(phi, epDataset);
            params.col(ep) = mleCalculator.getMeanParameters();
        }
        catch (...)
        {
            std::cout << "Error processing trajectory of episode: " << ep << std::endl;
        }
    }

    mu = arma::mean(params, 1);
    Sigma = arma::cov(params.t());
}

arma::mat MLEDistributionLinear::getParameters()
{
    return params;
}

ParametricNormal MLEDistributionLinear::getDistribution()
{
    return ParametricNormal(mu, Sigma);
}




}
