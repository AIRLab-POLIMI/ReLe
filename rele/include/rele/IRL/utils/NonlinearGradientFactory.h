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

#ifndef INCLUDE_RELE_IRL_UTILS_NONLINEARGRADIENTFACTORY_H_
#define INCLUDE_RELE_IRL_UTILS_NONLINEARGRADIENTFACTORY_H_

#include "rele/IRL/utils/gradient_nonlinear/NonlinearReinforceCalculator.h"
#include "rele/IRL/utils/gradient_nonlinear/NonlinearGPOMDPCalculator.h"
#include "rele/IRL/utils/gradient_nonlinear/NonlinearENACCalculator.h"
#include "rele/IRL/utils/gradient_nonlinear/NonlinearNaturalGradientCalculator.h"
#include "rele/IRL/utils/IrlGradType.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearGradientFactory
{
public:
    static NonlinearGradientCalculator<ActionC, StateC>* build(IrlGrad type,
            ParametricRegressor& rewardFunc,
            Dataset<ActionC,StateC>& data,
            DifferentiablePolicy<ActionC,StateC>& policy,
            double gamma)
    {

        switch(type)
        {
        case IrlGrad::REINFORCE:
            return new NonlinearReinforceCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlGrad::REINFORCE_BASELINE:
            return new NonlinearReinforceBaseCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlGrad::GPOMDP:
            return new NonlinearGPOMDPCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlGrad::GPOMDP_BASELINE:
            return new NonlinearGPOMDPBaseCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlGrad::ENAC:
            return new NonlinearENACCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlGrad::ENAC_BASELINE:
            return new NonlinearENACBaseCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma);

        default:
            return buildNatural(type, rewardFunc, data, policy, gamma);

        }
    }

private:
    static NonlinearGradientCalculator<ActionC, StateC>* buildNatural(IrlGrad type,
            ParametricRegressor& rewardFunc,
            Dataset<ActionC,StateC>& data,
            DifferentiablePolicy<ActionC,StateC>& policy,
            double gamma)
    {
        switch(type)
        {
        case IrlGrad::NATURAL_REINFORCE:
            return new NonlinearNaturalGradientCalculator<ActionC, StateC, NonlinearReinforceCalculator<ActionC, StateC>>(rewardFunc, data, policy, gamma);

        case IrlGrad::NATURAL_REINFORCE_BASELINE:
            return new NonlinearNaturalGradientCalculator<ActionC, StateC, NonlinearReinforceBaseCalculator<ActionC, StateC>>(rewardFunc, data, policy, gamma);

        case IrlGrad::NATURAL_GPOMDP:
            return new NonlinearNaturalGradientCalculator<ActionC, StateC, NonlinearGPOMDPCalculator<ActionC, StateC>>(rewardFunc, data, policy, gamma);

        case IrlGrad::NATURAL_GPOMDP_BASELINE:
            return new NonlinearNaturalGradientCalculator<ActionC, StateC, NonlinearGPOMDPBaseCalculator<ActionC, StateC>>(rewardFunc, data, policy, gamma);

        default:
            return nullptr;
        }
    }


};

}



#endif /* INCLUDE_RELE_IRL_UTILS_NONLINEARGRADIENTFACTORY_H_ */
