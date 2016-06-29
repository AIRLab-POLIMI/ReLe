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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EGIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EGIRL_H_

#include "rele/IRL/utils/EpisodicGradientCalculatorFactory.h"
#include "EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class EGIRL: public EpisodicLinearIRLAlgorithm<ActionC, StateC>
{
public:
    EGIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, DifferentiableDistribution& dist,
          LinearApproximator& rewardf, double gamma, IrlEpGrad type)
        : EpisodicLinearIRLAlgorithm<ActionC, StateC>(data, rewardf, gamma), type(type)
    {
        gradientCalculator = EpisodicGradientCalculatorFactory<ActionC, StateC>::build(type, theta, this->phiBar, dist, gamma);
    }

    virtual double objFunction(const arma::vec& xSimplex, arma::vec& df) override
    {
        ++this->nbFunEvals;

        // reconstruct parameters
        arma::vec&& x = this->simplex.reconstruct(xSimplex);

        // dispatch the right call
        arma::vec gradient = gradientCalculator->computeGradient(x);
        arma::mat dGradient = gradientCalculator->getGradientDiff();

        // compute objective function and derivative
        double f = arma::as_scalar(gradient.t() * gradient);
        arma::vec df_full = 2.0 * dGradient.t() * gradient;

        //compute the derivative wrt active features and simplex
        df = this->simplex.diff(df_full);

        return f;
    }


    virtual ~EGIRL()
    {

    }

protected:
    IrlEpGrad type;
    GradientCalculator<ActionC, StateC>* gradientCalculator;
};



}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EGIRL_H_ */
