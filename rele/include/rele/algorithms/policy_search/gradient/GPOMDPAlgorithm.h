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

#ifndef GPOMDPALGORITHM_H_
#define GPOMDPALGORITHM_H_

#include "rele/algorithms/policy_search/gradient/PolicyGradientAlgorithm.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// GPOMDP GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////

template<class ActionC, class StateC>
class GPOMDPAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{
    USE_PGA_MEMBERS

public:
    enum class BaseLineType
    {
        MULTI, SINGLE
    };


    GPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                    unsigned int nbEpisodes, unsigned int nbSteps, GradientStep& stepL,
                    BaseLineType btype, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, true, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(btype)
    {
        stepCount = 0;
    }

    GPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                    unsigned int nbEpisodes, unsigned int nbSteps, GradientStep& stepL,
                    int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, false, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(BaseLineType::SINGLE)
    {
        stepCount = 0;
    }


    GPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                    unsigned int nbEpisodes, unsigned int nbSteps, GradientStep& stepL,
                    BaseLineType btype, RewardTransformation& rewardt) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, rewardt, true),
        maxStepsPerEpisode(nbSteps),
        bType(btype)
    {
        stepCount = 0;
    }

    GPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                    unsigned int nbEpisodes, unsigned int nbSteps, GradientStep& stepL,
                    RewardTransformation& rewardt) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, rewardt, false),
        maxStepsPerEpisode(nbSteps),
        bType(BaseLineType::SINGLE)
    {
        stepCount = 0;
    }

    virtual ~GPOMDPAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init() override
    {
        unsigned int dp = policy.getParametersSize();
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdLogPi.assign(nbEpisodesToEvalPolicy,arma::vec(dp));
        sumdLogPi.set_size(dp);

        // variables for baseline settings
        baseline_num.zeros(dp,maxStepsPerEpisode);
        baseline_den.zeros(dp,maxStepsPerEpisode);
        baseline_num_single.zeros(dp);
        baseline_den_single.zeros(dp);
        baseline_num1_single.zeros(dp);
        baseline_num2_single.zeros(dp);
        episodeStepReward.zeros(nbEpisodesToEvalPolicy,maxStepsPerEpisode);
        sumGradLog.zeros(nbEpisodesToEvalPolicy,maxStepsPerEpisode, dp);
        episodeLength.zeros(nbEpisodesToEvalPolicy);
    }

    virtual void initializeVariables() override
    {
        sumdLogPi.zeros();
        stepCount = 0;
        baseline_num1_single.zeros();
        baseline_num2_single.zeros();
    }

    virtual void updateStep(const Reward& reward) override
    {
        RewardTransformation& rTr = *rewardTr;
        double reward_t = rTr(reward);

        //int dp = policy.getParametersSize();

        arma::vec grad = policy.difflog(currentState, currentAction);
        sumdLogPi += grad;

        // store the basic elements used to compute the gradient
        episodeStepReward(epiCount, stepCount) = df * reward_t;
        sumGradLog.tube(epiCount,stepCount) = sumdLogPi;

        // compute the baseline
        if (useBaseline && bType == BaseLineType::MULTI)
        {
            baseline_num.col(stepCount) += df * reward_t * sumdLogPi % sumdLogPi;
            baseline_den.col(stepCount) += sumdLogPi % sumdLogPi;
        }
        else if (useBaseline && bType == BaseLineType::SINGLE)
        {
            baseline_num1_single += df * reward_t * sumdLogPi;
            baseline_num2_single += sumdLogPi;
        }

        stepCount++;
    }

    virtual void updateAtEpisodeEnd() override
    {
        episodeLength(epiCount) = stepCount;

        // compute the baseline
        if (useBaseline && bType == BaseLineType::SINGLE)
        {
            baseline_num_single += baseline_num1_single % baseline_num2_single;
            baseline_den_single += baseline_num2_single % baseline_num2_single;
        }
    }

    virtual void updatePolicy() override
    {

        //std::cerr<< "#############################" << std::endl;
        //std::cerr << arma::sum(episodeStepReward, 1).t() << std::endl;
        //std::cerr << "#############################" << std::endl;
        //std::cerr << episodeStepReward << std::endl;
        //std::cerr << "#############################" << std::endl;
        int nbParams = policy.getParametersSize();
        arma::vec gradient(nbParams, arma::fill::zeros);

        // compute the gradient
        if(useBaseline)
        {
            if (bType == BaseLineType::MULTI)
            {
                for (int ep = 0; ep < nbEpisodesToEvalPolicy; ep++)
                {
                    for (int t = 0; t < episodeLength(ep); t++)
                    {
                        arma::vec baseline = baseline_num.col(t) / baseline_den.col(t);
                        baseline(arma::find_nonfinite(baseline)).zeros();
                        const arma::vec& sumGradLog_ep_t = sumGradLog.tube(ep,t);
                        //std::cerr << "t = " << t << std::endl;
                        //std::cerr << "b = " << baseline.t() << std::endl;
                        //std::cerr << "g = " << ((episodeStepReward(ep,t) - baseline) % sumGradLog_ep_t).t() << std::endl;
                        gradient += (episodeStepReward(ep,t) - baseline) % sumGradLog_ep_t;
                    }
                }

            }
            else
            {
                arma::vec baseline = baseline_num_single / baseline_den_single;
                baseline(arma::find_nonfinite(baseline)).zeros();

                for (int ep = 0; ep < nbEpisodesToEvalPolicy; ep++)
                {
                    for (int t = 0; t < episodeLength(ep); t++)
                    {
                        const arma::vec& sumGradLog_ep_t = sumGradLog.tube(ep,t);
                        gradient += (episodeStepReward(ep,t) - baseline) % sumGradLog_ep_t;
                    }
                }
            }
        }
        else
        {
            for (int ep = 0; ep < nbEpisodesToEvalPolicy; ep++)
            {
                for (int t = 0; t < episodeLength(ep); t++)
                {
                    const arma::vec& sumGradLog_ep_t = sumGradLog.tube(ep,t);
                    gradient += episodeStepReward(ep,t) * sumGradLog_ep_t;
                }
            }
        }

        // compute mean value
        if (task.gamma == 1.0)
            gradient /= totstep;
        else
            gradient /= nbEpisodesToEvalPolicy;

        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdLogPi;
        currentItStats->estimated_gradient = gradient;
        //---

        arma::vec newvalues = policy.getParameters() + stepLength(gradient);
        policy.setParameters(newvalues);


        baseline_num_single.zeros();
        baseline_den_single.zeros();
        baseline_num1_single.zeros();
        baseline_num2_single.zeros();
        baseline_den.zeros();
        baseline_num.zeros();
        episodeStepReward.zeros();

    }

protected:
    std::vector<arma::vec> history_sumdLogPi;
    arma::vec sumdLogPi;
    arma::mat episodeStepReward;
    arma::cube sumGradLog;
    arma::ivec episodeLength;

    unsigned int maxStepsPerEpisode, stepCount;

    arma::mat baseline_num, baseline_den;
    arma::vec baseline_num1_single, baseline_num2_single, baseline_num_single, baseline_den_single;
    BaseLineType bType;
};

}// end namespace ReLe

#endif //GPOMDPALGORITHM_H_
