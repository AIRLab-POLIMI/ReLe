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

#ifndef REINFORCEALGORITHM_H_
#define REINFORCEALGORITHM_H_

#include "rele/algorithms/policy_search/gradient/PolicyGradientAlgorithm.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// REINFORCE GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////
template<class ActionC, class StateC>
class REINFORCEAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{

    USE_PGA_MEMBERS

public:
    REINFORCEAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                       unsigned int nbEpisodes, GradientStep& stepL,
                       bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj)
    {
    }

    REINFORCEAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                       unsigned int nbEpisodes, GradientStep& stepL,
                       RewardTransformation& reward_tr,
                       bool baseline = true) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, reward_tr, baseline)
    {
    }

    virtual ~REINFORCEAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init() override
    {
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(policy.getParametersSize()));

        baseline_den.zeros(policy.getParametersSize());
        baseline_num.zeros(policy.getParametersSize());

        sumdlogpi.set_size(policy.getParametersSize());
    }

    virtual void initializeVariables() override
    {
        sumdlogpi.zeros();
    }

    virtual void updateStep(const Reward& reward) override
    {
        arma::vec grad = policy.difflog(currentState, currentAction);
        sumdlogpi += grad;
    }

    virtual void updateAtEpisodeEnd() override
    {
        if(useBaseline)
        {
            baseline_num += Jep * sumdlogpi % sumdlogpi;
            baseline_den += sumdlogpi % sumdlogpi;
        }

        history_sumdlogpi[epiCount] = sumdlogpi;
    }

    virtual void updatePolicy() override
    {
        int nbParams = policy.getParametersSize();
        arma::vec gradient(nbParams, arma::fill::zeros);
        // In the previous loop I have computed the baseline
        if(useBaseline)
        {
            arma::vec baseline = baseline_num / baseline_den;
            baseline(arma::find_nonfinite(baseline)).zeros();

            for (int ep = 0; ep < nbEpisodesToEvalPolicy; ep++)
            {
                gradient += (history_J[ep] - baseline) % history_sumdlogpi[ep];
            }

        }
        else
        {
            for (int ep = 0; ep < nbEpisodesToEvalPolicy; ep++)
            {
                gradient += history_J[ep] * history_sumdlogpi[ep];
            }
        }



        // compute mean value
        if (task.gamma == 1.0)
            gradient /= totstep;
        else
            gradient /= nbEpisodesToEvalPolicy;

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = gradient;
        //---


        arma::vec newvalues = policy.getParameters() + stepLength(gradient);
        policy.setParameters(newvalues);

        if(useBaseline)
        {
            baseline_den.zeros();
            baseline_num.zeros();
        }
    }

protected:

    arma::vec sumdlogpi, baseline_den, baseline_num;
    std::vector<arma::vec> history_sumdlogpi;
};

}// end namespace ReLe

#endif //REINFORCEALGORITHM_H_
