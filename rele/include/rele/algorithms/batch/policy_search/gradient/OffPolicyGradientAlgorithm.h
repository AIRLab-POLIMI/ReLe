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

#include "rele/algorithms/step_rules/GradientStep.h"
#include "rele/core/BatchAgent.h"
#include "rele/algorithms/batch/policy_search/gradient/OffGradientCalculatorFactory.h"

namespace ReLe
{

template<class ActionC, class StateC>
class OffPolicyGradientAlgorithm: public BatchAgent<ActionC, StateC>
{
public:
    OffPolicyGradientAlgorithm(OffGradType type,
                               Policy<ActionC, StateC>& behaviour,
                               DifferentiablePolicy<ActionC, StateC>& policy,
                               GradientStep& stepRule, RewardTransformation* rewardf) :
        type(type), behaviour(behaviour), policy(policy),
        stepRule(stepRule), rewardf(rewardf)
    {
        calculator = nullptr;
    }

    virtual void init(Dataset<ActionC, StateC>& data, double gamma) override
    {
        if(calculator)
            delete calculator;

        this->gamma = gamma;
        calculator = OffGradientCalculatorFactory<ActionC, StateC>::build(type, *rewardf, data, policy, behaviour, this->gamma);
    }

    virtual void step() override
    {
        arma::vec gradient = calculator->computeGradient();

        // compute step size
        unsigned int dp = gradient.n_elem;
        arma::mat eMetric = arma::eye(dp,dp);
        arma::vec step_size = stepRule.stepLength(gradient, eMetric);

        // Update policy
        arma::vec newvalues = policy.getParameters() + gradient * step_size;
        policy.setParameters(newvalues);
    }

    virtual Policy<ActionC, StateC>* getPolicy() override
    {
        return &policy;
    }

    virtual ~OffPolicyGradientAlgorithm()
    {

    }

private:
    OffGradType type;
    OffGradientCalculator<ActionC, StateC>* calculator;
    RewardTransformation* rewardf;
    Policy<ActionC, StateC>& behaviour;
    DifferentiablePolicy<ActionC, StateC>& policy;
    GradientStep& stepRule;


};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_OFFPOLICYGRADIENTALGORITHM_H_ */
