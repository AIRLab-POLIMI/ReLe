/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALGPOMDP_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALGPOMDP_H_

#include "policy_search/gradient/HierarchicalPolicyGradient.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// GPOMDP GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////

template<class ActionC, class StateC>
class HierarchicalGPOMDPAlgorithm: public HierarchicalPolicyGradient<ActionC, StateC>
{
    USE_HPGA_MEMBERS


public:
    enum BaseLineType { MULTI, SINGLE };


    HierarchicalGPOMDPAlgorithm(DifferentiableOption<ActionC, StateC>& rootOption,
                                unsigned int nbEpisodes, unsigned int nbSteps, StepRule& stepL,
                                BaseLineType btype, int reward_obj = 0) :
        HierarchicalPolicyGradient<ActionC, StateC>(rootOption, nbEpisodes, stepL, true, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(btype)
    {
        stepCount = 0;
    }

    HierarchicalGPOMDPAlgorithm(DifferentiableOption<ActionC, StateC>& rootOption,
                                unsigned int nbEpisodes, unsigned int nbSteps, StepRule& stepL,
                                int reward_obj = 0) :
        HierarchicalPolicyGradient<ActionC, StateC>(rootOption, nbEpisodes, stepL, false, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(BaseLineType::SINGLE)
    {
        stepCount = 0;
    }


    HierarchicalGPOMDPAlgorithm(DifferentiableOption<ActionC, StateC>& rootOption,
                                unsigned int nbEpisodes, unsigned int nbSteps, StepRule& stepL,
                                BaseLineType btype, RewardTransformation& rewardt) :
        HierarchicalPolicyGradient<ActionC, StateC>(rootOption, nbEpisodes, stepL, rewardt, true),
        maxStepsPerEpisode(nbSteps),
        bType(btype)
    {
        stepCount = 0;
    }

    HierarchicalGPOMDPAlgorithm(DifferentiableOption<ActionC, StateC>& rootOption,
                                unsigned int nbEpisodes, unsigned int nbSteps, StepRule& stepL,
                                RewardTransformation& rewardt) :
        HierarchicalPolicyGradient<ActionC, StateC>(rootOption, nbEpisodes, stepL, rewardt, false),
        maxStepsPerEpisode(nbSteps),
        bType(BaseLineType::SINGLE)
    {
        stepCount = 0;
    }

    virtual ~HierarchicalGPOMDPAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init()
    {
        unsigned int dp = this->getPolicy().getParametersSize();
        HierarchicalPolicyGradient<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(dp));
        sumdlogpi.set_size(dp);

        // variables for baseline settings
        baseline_num.zeros(dp,maxStepsPerEpisode);
        baseline_den.zeros(dp,maxStepsPerEpisode);
        baseline_num_single.zeros(dp);
        baseline_den_single.zeros(dp);
        baseline_num1_single.zeros(dp);
        baseline_num2_single.zeros(dp);
        reward_EpStep.zeros(nbEpisodesToEvalPolicy,maxStepsPerEpisode);
        sumGradLog_CompEpStep.zeros(dp,nbEpisodesToEvalPolicy,maxStepsPerEpisode);
        maxsteps_Ep.zeros(nbEpisodesToEvalPolicy);
    }

    virtual void initializeVariables()
    {
        sumdlogpi.zeros();
        stepCount = 0;
        baseline_num1_single.zeros();
        baseline_num2_single.zeros();
    }

    virtual void updateStep(const Reward& reward)
    {
        RewardTransformation& rTr = *rewardTr;

        int dp = this->getPolicy().getParametersSize();

        arma::vec grad = this->getPolicy().difflog(currentState, currentAction);
        sumdlogpi += grad;

        // store the basic elements used to compute the gradient
        reward_EpStep(epiCount, stepCount) = df * rTr(reward);

        for (int p = 0; p < dp; ++p)
        {
            sumGradLog_CompEpStep(p,epiCount,stepCount) = sumdlogpi(p);
        }
        //        std::cout << sumdlogpi.t();

        // compute the baseline

        if (useBaseline && bType == BaseLineType::MULTI)
        {
            for (int p = 0; p < dp; ++p)
            {
                baseline_num(p,stepCount) += df * rTr(reward) * sumdlogpi(p) * sumdlogpi(p);
            }

            for (int p = 0; p < dp; ++p)
            {
                baseline_den(p,stepCount) += sumdlogpi(p) * sumdlogpi(p);
            }
        }
        else if (useBaseline && bType == BaseLineType::SINGLE)
        {
            for (int p = 0; p < dp; ++p)
            {
                baseline_num1_single(p) += df * rTr(reward) * sumdlogpi(p);
                baseline_num2_single(p) += sumdlogpi(p);
            }
        }

        stepCount++;
    }

    virtual void updateAtEpisodeEnd()
    {
        maxsteps_Ep(epiCount) = stepCount;

        // compute the baseline

        int nbParams = this->getPolicy().getParametersSize();
        for (int p = 0; p < nbParams; ++p)
        {
            baseline_num_single(p) += baseline_num1_single(p) * baseline_num2_single(p);
            baseline_den_single(p) += baseline_num2_single(p) * baseline_num2_single(p);
        }
    }

    virtual void updatePolicy()
    {
        int nbParams = this->getPolicy().getParametersSize();
        arma::vec gradient(nbParams, arma::fill::zeros);
        // compute the gradient

        if (bType == BaseLineType::MULTI)
        {
            for (int p = 0; p < nbParams; ++p)
            {
                for (int ep = 0; ep < nbEpisodesToEvalPolicy; ++ep)
                {
                    for (int t = 0, te = maxsteps_Ep(ep); t < te; ++t)
                    {

                        double baseline = (useBaseline == true && baseline_den(p,t) != 0) ? baseline_num(p,t) / baseline_den(p,t) : 0;

                        gradient[p] += (reward_EpStep(ep,t) - baseline) * sumGradLog_CompEpStep(p,ep,t);
                    }
                }
            }
        }
        else
        {
            // compute the gradient
            for (int p = 0; p < nbParams; ++p)
            {
                double baseline =  (useBaseline == true && baseline_den_single(p) != 0) ? baseline_num_single(p) / baseline_den_single(p) : 0;

                for (int ep = 0; ep < nbEpisodesToEvalPolicy; ++ep)
                {
                    for (int t = 0, te = maxsteps_Ep(ep); t < te; ++t)
                    {
                        gradient[p] += (reward_EpStep(ep,t) - baseline) * sumGradLog_CompEpStep(p,ep,t);
                    }
                }
            }
        }

        // compute mean value
        gradient /= nbEpisodesToEvalPolicy;

        //--- Compute learning step
        arma::mat eMetric = arma::eye(nbParams,nbParams);
        arma::vec step_size = stepLength.stepLength(gradient, eMetric);
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

        for (int p = 0; p < nbParams; ++p)
        {
            baseline_num_single(p) = 0.0;
            baseline_den_single(p) = 0.0;
            baseline_num1_single(p) = 0.0;
            baseline_num2_single(p) = 0.0;
            for (int t = 0; t < maxStepsPerEpisode; ++t)
            {
                baseline_den(p,t) = 0;
                baseline_num(p,t) = 0;
            }
        }
    }

protected:
    std::vector<arma::vec> history_sumdlogpi;
    arma::vec sumdlogpi;
    arma::mat reward_EpStep;
    arma::cube sumGradLog_CompEpStep;
    arma::ivec maxsteps_Ep;

    unsigned int maxStepsPerEpisode, stepCount;

    arma::mat baseline_num, baseline_den;
    arma::vec baseline_num1_single, baseline_num2_single, baseline_num_single, baseline_den_single;
    BaseLineType bType;
};

}// end namespace ReLe



#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALGPOMDP_H_ */
