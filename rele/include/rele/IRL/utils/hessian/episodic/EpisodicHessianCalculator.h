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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_EPISODIC_EPISODICHESSIANCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_EPISODIC_EPISODICHESSIANCALCULATOR_H_

#include "rele/IRL/utils/hessian/HessianCalculator.h"


namespace ReLe
{

template<class ActionC, class StateC>
class EpisodicHessianCalculator : public HessianCalculator<ActionC, StateC>
{
public:
    EpisodicHessianCalculator(const arma::mat& theta,
                              const arma::mat& phi,
                              DifferentiableDistribution& dist,
                              double gamma):
        HessianCalculator<ActionC, StateC>(theta.n_rows, phi.n_rows, gamma),
        theta(theta), phi(phi), dist(dist)
    {

    }


    virtual ~EpisodicHessianCalculator()
    {

    }


protected:
    const arma::mat& theta;
    const arma::mat& phi;
    DifferentiableDistribution& dist;

};

}

#define USING_EPISODIC_H_CALCULATORS_MEMBERS(ActionC, StateC) \
	typedef EpisodicHessianCalculator<ActionC, StateC> Base; \
	using Base::gamma; \
	using Base::theta; \
	using Base::phi; \
	using Base::dist;

#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_EPISODIC_EPISODICHESSIANCALCULATOR_H_ */
