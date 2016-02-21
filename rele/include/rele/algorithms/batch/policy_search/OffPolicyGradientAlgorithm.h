/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_OFFPOLICYGRADIENTALGORITHM_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_OFFPOLICYGRADIENTALGORITHM_H_

#include "rele/core/BatchAgent.h"
#include "rele/algorithms/batch/policy_search/gradient/OffGradientCalculatorFactory.h"

namespace ReLe
{

template<class ActionC, class StateC>
class OffPolicyGradientAlgorithm: public BatchAgent<FiniteAction, StateC>
{
public:
    OffPolicyGradientAlgorithm(double gamma, OffGradientType type,
                               Policy<ActionC, StateC>& behaviour, DifferentiablePolicy<ActionC, StateC>& policy,
                               StepRule& stepL, RewardTransformation* rewardf) :
        BatchAgent<ActionC, StateC>(gamma), type(type), behaviour(behaviour), policy(policy),
        stepL(stepL), rewardf(rewardf)
    {
        calculator = nullptr;
    }

    virtual void init(Dataset<FiniteAction, StateC>& data) override
    {
        if(calculator)
            delete calculator;

        calculator = new OffGradientCalculatorFactory<ActionC, StateC>::build(type, *rewardf, data, behaviour, policy, this->gamma);
    }

    virtual void step() override
    {
    	arma::vec gradient = calculator->computeGradient();

    	// compute step size
        arma::mat eMetric = arma::eye(dp,dp);
        arma::vec step_size = stepRule.stepLength(gradient, eMetric);

        // Update policy
        arma::vec newvalues = target.getParameters() + gradient * step_size;
        target.setParameters(newvalues);
    }

    virtual ~OffPolicyGradientAlgorithm()
    {

    }

private:
    OffGradientType type;
    OffGradientCalculator<ActionC, StateC>* calculator;
    RewardTransformation* rewardf;
    Policy<ActionC, StateC>& behaviour;
    DifferentiablePolicy<ActionC, StateC>& policy;
    StepRule& stepL;


};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_OFFPOLICYGRADIENTALGORITHM_H_ */
