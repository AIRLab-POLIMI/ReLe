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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_MLEDISTRIBUTION_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_MLEDISTRIBUTION_H_

#include "rele/core/Transition.h"
#include "rele/policy/utils/MLE.h"
#include "rele/statistics/DifferentiableNormals.h"

namespace ReLe
{

template<class ActionC, class StateC>
class MLEDistribution
{
public:
    MLEDistribution(DifferentiablePolicy<ActionC, StateC>& policy)
        : policy(policy)
    {

    }

    void compute(const Dataset<ActionC, StateC>& data)
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int n = data.size();

        params.zeros(dp, n);

        //Compute policy MLE for each element
        for (unsigned int ep = 0; ep < data.size(); ep++)
        {
            Dataset<ActionC, StateC> epDataset;
            epDataset.push_back(data[ep]);
            MLE<ActionC, StateC> mleCalculator(policy, epDataset);
            double thetaP = mleCalculator.compute();
            params.col(ep) = policy.getParameters();

            std::cout << thetaP << std::endl;
        }

        mu = arma::mean(params, 1);
        //TODO [IMPORTANT] LEVAMI
        Sigma = arma::cov(params.t());
        //Sigma = arma::eye(dp, dp)*1e-3;

        std::cout << mu.t() << std::endl;
        std::cout << Sigma << std::endl;
        std::cout << arma::rank(Sigma) << std::endl;

        /*if(arma::rank(Sigma) < dp)
        {
            Sigma += arma::eye(dp, dp)*1e-6;
        }*/

    }

    arma::mat getParameters()
    {
        return params;
    }

    ParametricNormal getDistribution()
    {
        return ParametricNormal(mu, Sigma);
    }

private:
    arma::mat params;
    DifferentiablePolicy<ActionC, StateC>& policy;
    arma::vec mu;
    arma::mat Sigma;
};



}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_MLEDISTRIBUTION_H_ */
