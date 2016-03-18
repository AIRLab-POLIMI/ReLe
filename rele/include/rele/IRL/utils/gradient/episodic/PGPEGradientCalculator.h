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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_EPISODIC_PGPEGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_EPISODIC_PGPEGRADIENTCALCULATOR_H_

#include "rele/IRL/utils/gradient/episodic/EpisodicGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PGPEGradientCalculator : public EpisodicGradientCalculator<ActionC, StateC>
{
protected:
    USING_EPISODIC_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    PGPEGradientCalculator(const arma::mat& theta,
                           const arma::mat& phi,
                           DifferentiableDistribution& dist,
                           double gamma)
        : EpisodicGradientCalculator(theta, phi, dist, gamma)
    {

    }

    virtual ~PGPEGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
    	unsigned int N = theta.n_cols;
    	arma::mat sumGradLog = theta.each_col([&](const vec& b){ return dist.difflog(theta); });
    	return sumGradLog*phi.t()/N;
    }

};

template<class ActionC, class StateC>
class PGPEBaseGradientCalculator : public EpisodicGradientCalculator<ActionC, StateC>
{
protected:
    USING_EPISODIC_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    PGPEBaseGradientCalculator(const arma::mat& theta,
                               const arma::mat& phi,
                               DifferentiableDistribution& dist,
                               double gamma)
        : EpisodicGradientCalculator(theta, phi, dist, gamma)
    {

    }

    virtual ~PGPEBaseGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {



    }


};

}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_EPISODIC_PGPEGRADIENTCALCULATOR_H_ */
