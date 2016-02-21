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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_OFFGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_OFFGRADIENTCALCULATOR_H_


namespace ReLe
{

template<class ActionC, class StateC>
class OffGradientCalculator
{
public:
    OffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                          Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                          double gamma) :
        rewardf(rewardf), data(data), behaviour(behaviour), policy(policy), gamma(gamma)

    {

    }

    virtual arma::vec computeGradient() = 0;

    virtual ~OffGradientCalculator()
    {

    }

protected:
    RewardTransformation& rewardf;
    Dataset<ActionC,StateC>& data;
    Policy<ActionC,StateC>& behaviour;
    DifferentiablePolicy<ActionC,StateC>& policy;
    double gamma;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_OFFGRADIENTCALCULATOR_H_ */

