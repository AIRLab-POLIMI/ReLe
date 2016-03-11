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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_OFFGRADIENTCALCULATORFACTORY_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_OFFGRADIENTCALCULATORFACTORY_H_

#include "rele/algorithms/batch/policy_search/gradient/OffGradType.h"
#include "rele/algorithms/batch/policy_search/gradient/ReinforceOffGradientCalculator.h"
#include "rele/algorithms/batch/policy_search/gradient/GPOMDPOffGradientCalculator.h"
#include "rele/algorithms/batch/policy_search/gradient/SecondMomentOffGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class OffGradientCalculatorFactory
{

public:
    static OffGradientCalculator<ActionC, StateC>* build(OffGradType type,
            RewardTransformation& rewardf,
            Dataset<ActionC,StateC>& data,
            DifferentiablePolicy<ActionC,StateC>& policy,
            Policy<ActionC,StateC>& behaviour,
            double gamma)
    {

        switch(type)
        {
        case OffGradType::REINFORCE:
            return new ReinforceOffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma);

        case OffGradType::REINFORCE_BASELINE:
            return new ReinforceBaseOffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma);

        case OffGradType::GPOMDP:
            return new GPOMDPOffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma);

        case OffGradType::GPOMDP_BASELINE_SINGLE:
            return new GPOMDPSingleBaseOffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma);

        case OffGradType::GPOMDP_BASELINE_MULTY:
            return new GPOMDPMultyBaseOffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma);

        case OffGradType::SECOND_MOMENT:
            return new SecondMomentOffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma);

        default:
            return nullptr;

        }
    }


};

}


#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_OFFGRADIENTCALCULATORFACTORY_H_ */
