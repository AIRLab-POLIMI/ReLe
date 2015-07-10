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

#include "policy_search/gradient/PolicyGradientAlgorithm.h"

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
                       unsigned int nbEpisodes, StepRule& stepL,
                       bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj)
    {
    }

    REINFORCEAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                       unsigned int nbEpisodes, StepRule& stepL,
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
    virtual void init()
    {
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(policy.getParametersSize()));

        baseline_den.zeros(policy.getParametersSize());
        baseline_num.zeros(policy.getParametersSize());

        sumdlogpi.set_size(policy.getParametersSize());
    }

    virtual void initializeVariables()
    {
        sumdlogpi.zeros();
    }

    virtual void updateStep(const Reward& reward)
    {
        arma::vec grad = policy.difflog(currentState, currentAction);
        sumdlogpi += grad;
    }

    virtual void updateAtEpisodeEnd()
    {
        for (int p = 0; p < baseline_num.n_elem; ++p)
        {
            baseline_num[p] += Jep * sumdlogpi[p] * sumdlogpi[p];
            baseline_den[p] += sumdlogpi[p] * sumdlogpi[p];
            history_sumdlogpi[epiCount][p] = sumdlogpi[p];
        }
    }

    virtual void updatePolicy()
    {
        int nbParams = policy.getParametersSize();
        arma::vec gradient(nbParams, arma::fill::zeros);
        // In the previous loop I have computed the baseline
        for (int ep = 0; ep < nbEpisodesToEvalPolicy; ++ep)
        {
            for (int p = 0; p < nbParams; ++p)
            {
                double base_el = (useBaseline == false || baseline_den[p] == 0.0) ? 0 : baseline_num[p]/baseline_den[p];
                gradient[p] += history_sumdlogpi[ep][p] * (history_J[ep] - base_el);
            }
        }

        // compute mean value
        gradient /= nbEpisodesToEvalPolicy;

        //--- Compute learning step
        arma::vec step_size = stepLength.stepLength(gradient);
        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = gradient;
        currentItStats->stepLength = step_size;
        //---


        arma::vec newvalues = policy.getParameters() + gradient * step_size;
        policy.setParameters(newvalues);
        //        std::cout << "new_params: "  << newvalues.t();

        for (int i = 0; i < nbParams; ++i)
        {
            baseline_den[i] = 0;
            baseline_num[i] = 0;
        }
    }

protected:

    arma::vec sumdlogpi, baseline_den, baseline_num;
    std::vector<arma::vec> history_sumdlogpi;
};

}// end namespace ReLe

#endif //REINFORCEALGORITHM_H_
