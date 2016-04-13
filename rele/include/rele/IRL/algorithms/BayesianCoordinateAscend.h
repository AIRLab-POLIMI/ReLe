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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_

#include "rele/core/Transition.h"
#include "rele/statistics/inference/GaussianConjugatePrior.h"
#include "rele/policy/utils/MAP.h"

namespace ReLe
{

template<class ActionC, class StateC>
class BayesianCoordinateAscend
{
public:
    BayesianCoordinateAscend(const arma::mat& Sigma,
                             const ParametricNormal& prior,
                             DifferentiablePolicy<ActionC, StateC>& policy)
        : Sigma(Sigma), prior(prior), policy(policy),
		  thetaPrior(prior.getMean(), Sigma),
		  posterior(policy.getParametersSize())
    {

    }

    void compute(const Dataset<ActionC, StateC>& data)
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int n = data.size();

        params.zeros(dp, n);

        double eps = 1e-8;
        double posteriorP = -std::numeric_limits<double>::infinity();
        double oldPosteriorP;

        do
        {
            //Reset posterior probability
            oldPosteriorP = posteriorP;

            //Compute policy MAP for each element
			posteriorP = updateTheta(data);

            //Compute distribution posterior
			posteriorP += computePosterior();

            //Update theta prior
			computeThetaPrior();

        }
        while(posteriorP - oldPosteriorP > eps);
    }

    ParametricNormal getPosterior()
    {
        return posterior;
    }

protected:
	double updateTheta(const Dataset<ActionC, StateC>& data)
	{
		 double posteriorP = 0;

		//Compute policy MAP for each element
		for (unsigned int ep = 0; ep < data.size(); ep++)
		{
			Dataset<ActionC, StateC> epDataset;
			epDataset.push_back(data[ep]);
			MAP<ActionC, StateC> mapCalculator(policy, thetaPrior, epDataset);
			arma::vec theta_ep = params.col(ep);
			posteriorP += mapCalculator.compute(theta_ep);
			params.col(ep) = policy.getParameters();
		}

		return posteriorP;
	}

	double computePosterior()
	{
		//Compute distribution posterior
		posterior = GaussianConjugatePrior::compute(Sigma, prior, params);

		//compute posterior probability
		arma::vec omega = posterior.getMean();
		return std::log(posterior(omega));
	}

	void computeThetaPrior()
	{
		thetaPrior = ParametricNormal(posterior.getMean(), Sigma);
	}

private:
    arma::mat params;
    const arma::mat& Sigma;
    const ParametricNormal& prior;
    DifferentiablePolicy<ActionC, StateC>& policy;
    ParametricNormal thetaPrior;
    ParametricNormal posterior;


};


}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_ */
