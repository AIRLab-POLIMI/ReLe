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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARGPOMDPCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARGPOMDPCALCULATOR_H_

#include "rele/IRL/utils/gradient_nonlinear/NonlinearGradientCalculator.h"


namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearGPOMDPCalculator
{
public:
    NonlinearGPOMDPCalculator(Regressor& rewardFunc,
                              Dataset<ActionC,StateC>& data,
                              DifferentiablePolicy<ActionC,StateC>& policy,
                              double gamma) : NonlinearGradientCalculator(rewardFunc, data, policy, gamma)
    {

    }

    virtual ~NonlinearGPOMDPCalculator()
    {

    }
};

template<class ActionC, class StateC>
class NonlinearGPOMDPBaseCalculator
{
public:
    NonlinearGPOMDPBaseCalculator(Regressor& rewardFunc,
                                  Dataset<ActionC,StateC>& data,
                                  DifferentiablePolicy<ActionC,StateC>& policy,
                                  double gamma) : NonlinearGradientCalculator(rewardFunc, data, policy, gamma)
    {

    }

    virtual ~NonlinearGPOMDPBaseCalculator()
    {

    }
};


}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARGPOMDPCALCULATOR_H_ */
