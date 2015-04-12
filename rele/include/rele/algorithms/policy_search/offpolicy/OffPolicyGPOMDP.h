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

#ifndef OFFPOLICYGPOMDP_H_
#define OFFPOLICYGPOMDP_H_

#include "OffPolicyGradientAlgorithm.h"

namespace ReLe
{

//Templates needed to handle different action types
template<class StateC, class PolicyC, class PolicyC2>
double OffPolicyGpomdpIWWorker(const StateC& state, const FiniteAction& action, PolicyC& policy, PolicyC2& behav,
                               double& iwb, double& iwt)
{
    typename action_type<FiniteAction>::type_ref u = action.getActionN();

    double valb = behav(state,u);
    double valt = policy(state,u);

    iwt *= valt;
    iwb *= valb;

    return valt/valb;
}

template<class StateC, class ActionC, class PolicyC, class PolicyC2>
double OffPolicyGpomdpIWWorker(const StateC& state, const ActionC& action, PolicyC& policy, PolicyC2& behav,
                               double& iwb, double& iwt)
{
    double valb = behav(state,action);
    double valt = policy(state,action);

    iwt *= valt;
    iwb *= valb;

    return valt/valb;
}

template<class ActionC, class StateC>
class OffPolicyGPOMDP: public AbstractOffPolicyGradientAlgorithm<ActionC, StateC>
{

    USE_PUREOFFPGA_MEMBERS

public:
    enum BaseLineType { MULTI, SINGLE };

    OffPolicyGPOMDP(DifferentiablePolicy<ActionC, StateC>& target_pol,
                    Policy<ActionC, StateC>& behave_pol,
                    unsigned int nbEpisodes, unsigned int nbSamplesForJ, unsigned int nbSteps,
                    StepRule& stepL, BaseLineType btype, int reward_obj = 0) :
        AbstractOffPolicyGradientAlgorithm<ActionC, StateC>(target_pol, behave_pol, nbEpisodes, nbSamplesForJ, stepL, true, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(btype)
    {
    }

    OffPolicyGPOMDP(DifferentiablePolicy<ActionC, StateC>& target_pol,
                    Policy<ActionC, StateC>& behave_pol,
                    unsigned int nbEpisodes, unsigned int nbSamplesForJ, unsigned int nbSteps,
                    StepRule& stepL, int reward_obj = 0) :
        AbstractOffPolicyGradientAlgorithm<ActionC, StateC>(target_pol, behave_pol, nbEpisodes, nbSamplesForJ, stepL, false, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(BaseLineType::SINGLE)
    {
    }

    virtual ~OffPolicyGPOMDP()
    {
    }


    // Agent interface
protected:
    virtual void init()
    {
        unsigned int dp = target.getParametersSize();
        AbstractOffPolicyGradientAlgorithm<ActionC, StateC>::init();
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
        sumIWOverRun = 0.0;
        sumdlogpi.zeros();
        stepCount = 0;
        baseline_num1_single.zeros();
        baseline_num2_single.zeros();
    }

    virtual double updateStep(const Reward& reward)
    {

        int dp = policy.getParametersSize();

        //compute importance weight
        double currIW = OffPolicyReinforceIWWorker(
                            currentState, currentAction, target, behavioral, prodImpWeightB, prodImpWeightT);
        double IW     = prodImpWeightT / prodImpWeightB;
        sumIWOverRun += IW;
        IW_EpStep(epCounter,stepCount) = IW;

        //compute policy gradient
        arma::vec grad = diffLogWorker(currentState, currentAction, policy);
        sumdlogpi += grad;

        // store the basic elements used to compute the gradient
        reward_EpStep(epCounter, stepCount) = df * reward[rewardId];

        for (int p = 0; p < dp; ++p)
        {
            sumGradLog_CompEpStep(p,epCounter,stepCount) = sumdlogpi(p);
        }
        //        std::cout << sumdlogpi.t();

        // compute the baseline

        if (useBaseline && bType == BaseLineType::MULTI)
        {
            for (int p = 0; p < dp; ++p)
            {
                baseline_num(p,stepCount) += df * reward[rewardId] * IW * IW * sumdlogpi(p) * sumdlogpi(p);
                baseline_den(p,stepCount) += IW * IW * sumdlogpi(p) * sumdlogpi(p);
            }
        }
        else if (useBaseline && bType == BaseLineType::SINGLE)
        {
            for (int p = 0; p < dp; ++p)
            {
                baseline_num1_single(p) += df * reward[rewardId] * IW * sumdlogpi(p);
                baseline_num2_single(p) += IW * sumdlogpi(p);
            }
        }

        stepCount++;

        return currIW;
    }

    virtual void updateAtEpisodeEnd()
    {
        maxsteps_Ep(epiCount) = stepCount;

        // compute the baseline

        int nbParams = policy.getParametersSize();
        for (int p = 0; p < nbParams; ++p)
        {
            baseline_num_single(p) += baseline_num1_single(p) * baseline_num2_single(p);
            baseline_den_single(p) += baseline_num2_single(p) * baseline_num2_single(p);
        }

        // the weight used in the mean is the sum of the mean Importance Weights of each steps in the episode
        sumAvgIW += sumIWOverRun / stepCount;
    }

    virtual void updatePolicy()
    {
        int dp = policy.getParametersSize();
        arma::vec gradient(dp, arma::fill::zeros);
        // compute the gradient

        if (bType == BaseLineType::MULTI)
        {
            for (int p = 0; p < dp; ++p)
            {
                for (int ep = 0; ep < nbEpisodesToEvalPolicy; ++ep)
                {
                    for (int t = 0, te = maxsteps_Ep(ep); t < te; ++t)
                    {

                        double baseline = (useBaseline == true && baseline_den(p,t) != 0) ? baseline_num(p,t) / baseline_den(p,t) : 0;

                        gradient[p] += (reward_EpStep(ep,t) - baseline) * IW_EpStep(ep,t) * sumGradLog_CompEpStep(p,ep,t);
                    }
                }
            }
        }
        else
        {
            // compute the gradient
            for (int p = 0; p < dp; ++p)
            {
                double baseline =  (useBaseline == true && baseline_den_single(p) != 0) ? baseline_num_single(p) / baseline_den_single(p) : 0;

                for (int ep = 0; ep < nbEpisodesToEvalPolicy; ++ep)
                {
                    for (int t = 0, te = maxsteps_Ep(ep); t < te; ++t)
                    {
                        gradient[p] += (reward_EpStep(ep,t) - baseline) * IW_EpStep(ep,t) * sumGradLog_CompEpStep(p,ep,t);
                    }
                }
            }
        }

        // compute mean value
//        gradient /= nbEpisodesToEvalPolicy;
        gradient /= sumAvgIW;

        //--- Compute learning step
        arma::mat eMetric = arma::eye(dp,dp);
        arma::vec step_size = stepLength.stepLength(gradient, eMetric);
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

        for (int p = 0; p < dp; ++p)
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
    arma::vec sumdlogpi;
    std::vector<arma::vec> history_sumdlogpi;
    arma::mat reward_EpStep;
    arma::cube sumGradLog_CompEpStep;
    arma::ivec maxsteps_Ep;
    double prodImpWeightB, prodImpWeightT;

    unsigned int maxStepsPerEpisode, stepCount;

    arma::mat baseline_num, baseline_den;
    arma::vec baseline_num1_single, baseline_num2_single, baseline_num_single, baseline_den_single;
    BaseLineType bType;
};

} //end namespace

#endif //OFFPOLICYGPOMDP_H_
