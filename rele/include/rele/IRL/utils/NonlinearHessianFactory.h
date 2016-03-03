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

#ifndef INCLUDE_RELE_IRL_UTILS_NONLINEARHESSIANFACTORY_H_
#define INCLUDE_RELE_IRL_UTILS_NONLINEARHESSIANFACTORY_H_

#include "rele/IRL/utils/hessian_nonlinear/NonlinearHessianReinforce.h"
#include "rele/IRL/utils/hessian_nonlinear/NonlinearHessianGPOMDP.h"
#include "rele/IRL/utils/IrlGradType.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearHessianFactory
{
public:
    static NonlinearHessianCalculator<ActionC, StateC>* build(IrlHess type,
            ParametricRegressor& rewardFunc,
            Dataset<ActionC,StateC>& data,
            DifferentiablePolicy<ActionC,StateC>& policy,
            double gamma)
    {

        switch(type)
        {
        case IrlHess::REINFORCE:
            return new NonlinearHessianReinforce<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlHess::REINFORCE_BASELINE:
            return new NonlinearHessianReinforceBase<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlHess::GPOMDP:
            return new NonlinearHessianGPOMDP<ActionC, StateC>(rewardFunc, data, policy, gamma);

        case IrlHess::GPOMDP_BASELINE:
            return new NonlinearHessianGPOMDPBase<ActionC, StateC>(rewardFunc, data, policy, gamma);

        default:
            return nullptr;

        }
    }


};


}



#endif /* INCLUDE_RELE_IRL_UTILS_NONLINEARHESSIANFACTORY_H_ */
