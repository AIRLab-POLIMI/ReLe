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
          double penalization, double fraction = 0.5, unsigned int rewardIndex = 0)
        : MBPGA(behaviour, policy, stepRule, new IndexRT(rewardIndex), penalization, fraction)
    {
        deleteReward = true;
    }

    MBPGA(Policy<ActionC, StateC>& behaviour,
          DifferentiablePolicy<ActionC, StateC>& policy,
          GradientStep& stepRule,
          RewardTransformation* rewardf,
          double penalization, double fraction = 0.5) :
        behaviour(behaviour), policy(policy),
        stepRule(stepRule), rewardf(rewardf), penalization(penalization), fraction(fraction), deleteReward(false)
    {
        gCalculator = nullptr;
        gM2Calculator = nullptr;
        Jmean = 0;
    }

    virtual void init(Dataset<ActionC, StateC>& data, EnvironmentSettings& envSettings) override
    {
        if(gCalculator)
            delete gCalculator;

        if(gM2Calculator)
            delete gM2Calculator;

        this->gamma = envSettings.gamma;
        gCalculator = OffGradientCalculatorFactory<ActionC, StateC>::build(OffGradType::REINFORCE_BASELINE,
                      *rewardf, data, policy, behaviour, this->gamma);
        gM2Calculator = OffGradientCalculatorFactory<ActionC, StateC>::build(OffGradType::SECOND_MOMENT,
                        *rewardf, data, policy, behaviour, this->gamma);


        //Select indexes
        arma::uvec indexes = arma::linspace<arma::uvec>(0, data.size() - 1, data.size());
        arma::shuffle(indexes);

        unsigned int breakPoint = data.size()*fraction;

        gradientIndexes = indexes.rows(0, breakPoint - 1);
        arma::uvec meanIndexes = indexes.rows(breakPoint, data.size() - 1);

        computeJmean(meanIndexes, data, envSettings.gamma);
    }

    virtual void step() override
    {
        //Compute gradients
        arma::vec gradientJ = gCalculator->computeGradient(gradientIndexes);
        arma::vec gradientM2 = gM2Calculator->computeGradient(gradientIndexes);

        //Compute risk averse gradient
        arma::vec gradient = gradientJ - penalization * (gradientM2 - 2 * Jmean * gradientJ) / gradientIndexes.n_elem;

        // Update policy
        arma::vec newvalues = policy.getParameters() + stepRule(gradient);
        policy.setParameters(newvalues);
    }

    virtual Policy<ActionC, StateC>* getPolicy() override
    {
        return &policy;
    }

    virtual ~MBPGA()
    {
        if(deleteReward)
            delete rewardf;
    }


private:
    void computeJmean(const arma::uvec& indexes, const Dataset<ActionC, StateC>& data, double gamma)
    {
        Jmean = 0;

        for(auto indx : indexes)
        {
            auto& episode = data[indx];

            double Jep = 0;
            double df = 1.0;
            for(auto& tr : episode)
            {
                auto& rf = *rewardf;
                Jep += df*rf(tr.r);
                df *= gamma;
            }

            Jmean += Jep;

        }

        Jmean /= indexes.n_elem;
    }

private:
    OffGradientCalculator<ActionC, StateC>* gCalculator;
    OffGradientCalculator<ActionC, StateC>* gM2Calculator;
    RewardTransformation* rewardf;
    Policy<ActionC, StateC>& behaviour;
    DifferentiablePolicy<ActionC, StateC>& policy;

    arma::uvec gradientIndexes;
    double Jmean;

    GradientStep& stepRule;
    double penalization;
    double fraction;

    bool deleteReward;


};

}


#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_MBPGA_H_ */
