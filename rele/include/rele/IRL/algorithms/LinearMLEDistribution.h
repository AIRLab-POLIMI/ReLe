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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_LINEARMLEDISTRIBUTION_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_LINEARMLEDISTRIBUTION_H_

#include "rele/core/Transition.h"
#include "rele/policy/utils/MLE.h"
#include "rele/statistics/DifferentiableNormals.h"

namespace ReLe
{

class LinearMLEDistribution
{
public:
    LinearMLEDistribution(const Features& phi, const arma::mat& SigmaPolicy)
        : phi(phi), SigmaInv(SigmaPolicy.i())
    {

    }

    void compute(const Dataset<DenseAction, DenseState>& data)
    {
        unsigned int dp = phi.rows();
        unsigned int n = data.size();

        params.zeros(dp, n);

        //Compute policy MLE for each element
        for (unsigned int ep = 0; ep < data.size(); ep++)
        {

        	arma::mat A(dp, dp, arma::fill::zeros);
        	arma::vec b(dp, arma::fill::zeros);
        	for(auto& tr : data[ep])
        	{
        		const arma::vec& x = tr.x;
        		const arma::vec& u = tr.u;

        		arma::mat phiX = phi(x);
        		A += phiX*SigmaInv*phiX.t();
        		b += phiX*SigmaInv*u;
        	}

        	params.col(ep) = solve(A, b);
        }

        mu = arma::mean(params, 1);
        Sigma = arma::cov(params.t());

        std::cout << mu.t() << std::endl;
        std::cout << Sigma << std::endl;
        std::cout << arma::rank(Sigma) << std::endl;


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
    const Features& phi;
    arma::mat SigmaInv;

    arma::mat params;
    arma::vec mu;
    arma::mat Sigma;
};



}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_LINEARMLEDISTRIBUTION_H_ */
