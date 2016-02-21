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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_GPOMDPOFFGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_GPOMDPOFFGRADIENTCALCULATOR_H_

#include "rele/algorithms/batch/policy_search/gradient/OffGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class GPOMDPOffGradientCalculator
{
public:
    GPOMDPOffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                                Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma)
        : GradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma)

    {

    }

    virtual arma::vec computeGradient() override
    {

    }

    virtual ~GPOMDPOffGradientCalculator()
    {

    }

};

template<class ActionC, class StateC>
class GPOMDPBaseOffGradientCalculator
{
public:
    GPOMDPBaseOffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                                    Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                                    double gamma)
        : GradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma)

    {

    }

    virtual arma::vec computeGradient() override
    {

    }

    virtual ~GPOMDPBaseOffGradientCalculator()
    {

    }

};


}


#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_GPOMDPOFFGRADIENTCALCULATOR_H_ */
