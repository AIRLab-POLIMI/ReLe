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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_MBPGA_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_MBPGA_H_

#include "rele/algorithms/step_rules/GradientStep.h"
#include "rele/core/BatchAgent.h"
#include "rele/algorithms/batch/policy_search/gradient/OffGradientCalculatorFactory.h"

namespace ReLe
{

template<class ActionC, class StateC>
class MBPGA: public BatchAgent<ActionC, StateC>
{
public:
    MBPGA(Policy<ActionC, StateC>& behaviour,
          DifferentiablePolicy<ActionC, StateC>& policy,
          GradientStep& stepRule,
          RewardTransformation* rewardf,
          double penalization) :
        type(type), behaviour(behaviour), policy(policy),
        stepRule(stepRule), rewardf(rewardf), penalization(penalization)
    {
        gCalculator = nullptr;
        gM2Calculator = nullptr;
    }

    virtual void init(Dataset<ActionC, StateC>& data, double gamma) override
    {
        if(calculator)
            delete calculator;

        this->gamma = gamma;
        gCalculator = OffGradientCalculatorFactory<ActionC, StateC>::build(OffGradType::REINFORCE_BASELINE,
                      *rewardf, data, policy, behaviour, this->gamma);
        gM2Calculator = OffGradientCalculatorFactory<ActionC, StateC>::build(OffGradType::SECOND_MOMENT,
                        *rewardf, data, policy, behaviour, this->gamma);

        //TODO compute Jmean here
    }

    virtual void step() override
    {
        //Compute gradients
        arma::vec gradientJ = gCalculator->computeGradient();
        arma::vec gradientM2 = gM2Calculator->computeGradient();

        //compute mean
        //TODO implement
        arma::vec Jmean;
        unsigned int nbEpisodesperUpdate = 0;
        unsigned int nbIndipendentSamples = 0;

        //Compute risk averse gradient
        arma::vec gradient = gradientJ - penalization * (gradientM2 - 2 * Jmean * gradientJ) / (nbEpisodesperUpdate-nbIndipendentSamples);

        // compute step size
        unsigned int dp = gradient.n_elem;
        arma::vec step_size = stepRule.stepLength(gradient);

        // Update policy
        arma::vec newvalues = policy.getParameters() + gradient * step_size;
        policy.setParameters(newvalues);
    }

    virtual Policy<ActionC, StateC>* getPolicy() override
    {
        return &policy;
    }

    virtual ~MBPGA()
    {

    }

private:
    OffGradientCalculator<ActionC, StateC>* gCalculator;
    OffGradientCalculator<ActionC, StateC>* gM2Calculator;
    RewardTransformation* rewardf;
    Policy<ActionC, StateC>& behaviour;
    DifferentiablePolicy<ActionC, StateC>& policy;
    GradientStep& stepRule;
    double penalization;


};

}


#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_MBPGA_H_ */
