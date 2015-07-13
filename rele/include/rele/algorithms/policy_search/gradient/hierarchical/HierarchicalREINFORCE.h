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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALREINFORCE_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALREINFORCE_H_

#include "policy_search/gradient/HierarchicalPolicyGradient.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// REINFORCE GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////
template<class ActionC, class StateC>
class HierarchicalREINFORCE: public HierarchicalPolicyGradient<ActionC, StateC>
{

    USE_HPGA_MEMBERS

public:
    HierarchicalREINFORCE(DifferentiableOption<ActionC, StateC>& rootOption,
                          unsigned int nbEpisodes, StepRule& stepL,
                          bool baseline = true, int reward_obj = 0) :
        HierarchicalPolicyGradient<ActionC, StateC>(rootOption, nbEpisodes, stepL, baseline, reward_obj)
    {
    }

    HierarchicalREINFORCE(DifferentiableOption<ActionC, StateC>& rootOption,
                          unsigned int nbEpisodes, StepRule& stepL,
                          RewardTransformation& reward_tr,
                          bool baseline = true) :
        HierarchicalPolicyGradient<ActionC, StateC>(rootOption, nbEpisodes, stepL, reward_tr, baseline)
    {
    }

    virtual ~HierarchicalREINFORCE()
    {
    }

    // Agent interface
protected:
    virtual void init()
    {
        HierarchicalPolicyGradient<ActionC, StateC>::init();

        int parameterSize = this->getPolicy().getParametersSize();

        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(parameterSize));

        baseline_den.zeros(parameterSize);
        baseline_num.zeros(parameterSize);

        sumdlogpi.set_size(parameterSize);
    }

    virtual void initializeVariables()
    {
        sumdlogpi.zeros();
    }

    virtual void updateStep(const Reward& reward)
    {
        arma::vec grad = this->getPolicy().difflog(currentState, currentAction);
        sumdlogpi += grad;
    }

    virtual void updateAtEpisodeEnd()
    {
        if(useBaseline)
        {
            baseline_num += Jep * sumdlogpi % sumdlogpi;
            baseline_den += sumdlogpi % sumdlogpi;
        }

        history_sumdlogpi[epiCount] = sumdlogpi;
    }

    virtual void updatePolicy()
    {
        int nbParams = this->getPolicy().getParametersSize();
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


        arma::vec newvalues = this->getPolicy().getParameters() + gradient * step_size;
        this->getPolicy().setParameters(newvalues);
        //        std::cout << "new_params: "  << newvalues.t();

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


#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALREINFORCE_H_ */
