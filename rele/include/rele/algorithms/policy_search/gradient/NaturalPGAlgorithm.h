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

#ifndef NATURALPOLICYGRADIENTALGORITHM_H_
#define NATURALPOLICYGRADIENTALGORITHM_H_

#include "rele/algorithms/policy_search/gradient/PolicyGradientAlgorithm.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// NATURAL GPOMDP ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////
/**
 * A Natural Policy Gradient
 * Sham Kakade
 * NIPS
 * http://research.microsoft.com/en-us/um/people/skakade/papers/rl/natural.pdf
 */
template<class ActionC, class StateC>
class NaturalGPOMDPAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{
    USE_PGA_MEMBERS

public:
    NaturalGPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                           unsigned int nbEpisodes, unsigned int nbSteps, GradientStep& stepL,
                           bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj),
        maxStepsPerEpisode(nbSteps)
    {
        stepCount = 0;
    }

    NaturalGPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                           unsigned int nbEpisodes, unsigned int nbSteps, GradientStep& stepL,
                           RewardTransformation& reward_tr,
                           bool baseline = true) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, reward_tr, baseline),
        maxStepsPerEpisode(nbSteps)
    {
        stepCount = 0;
    }


    virtual ~NaturalGPOMDPAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init() override
    {
        unsigned int dp = policy.getParametersSize();
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(dp));
        sumdlogpi.set_size(dp);

        // variables for baseline settings
        baseline_num.zeros(dp,maxStepsPerEpisode);
        baseline_den.zeros(dp,maxStepsPerEpisode);
        reward_EpStep.zeros(nbEpisodesToEvalPolicy,maxStepsPerEpisode);
        sumGradLog_CompEpStep.zeros(dp,nbEpisodesToEvalPolicy,maxStepsPerEpisode);
        maxsteps_Ep.zeros(nbEpisodesToEvalPolicy);
        fisher.zeros(dp,dp);
        fisherEp.zeros(dp,dp);
    }

    virtual void initializeVariables() override
    {
        sumdlogpi.zeros();
        stepCount = 0;
        fisherEp.zeros();
    }

    virtual void updateStep(const Reward& reward) override
    {

        int dp = policy.getParametersSize();

        arma::vec grad = policy.difflog(currentState, currentAction);
        sumdlogpi += grad;

        fisherEp += grad * grad.t();

        // store the basic elements used to compute the gradient
        reward_EpStep(epiCount, stepCount) = df * rewardTr->operator ()(reward);

        for (int p = 0; p < dp; ++p)
        {
            sumGradLog_CompEpStep(p,epiCount,stepCount) = sumdlogpi(p);
        }

        // compute the baseline

        for (int p = 0; p < dp; ++p)
        {
            baseline_num(p,stepCount) += df * rewardTr->operator ()(reward) * sumdlogpi(p) * sumdlogpi(p);
        }

        for (int p = 0; p < dp; ++p)
        {
            baseline_den(p,stepCount) += sumdlogpi(p) * sumdlogpi(p);
        }

        stepCount++;
    }

    virtual void updateAtEpisodeEnd() override
    {
        maxsteps_Ep(epiCount) = stepCount;
        fisherEp /= stepCount;
        fisher += fisherEp;
    }

    virtual void updatePolicy() override
    {
        int nbParams = policy.getParametersSize();
        arma::vec gradient(nbParams, arma::fill::zeros);
        // compute the gradient (the gradient is estimated like GPOMDP)
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


        // compute mean value
        if (task.gamma == 1.0)
            gradient /= totstep;
        else
            gradient /= nbEpisodesToEvalPolicy;
        fisher /= nbEpisodesToEvalPolicy;

        //--- Compute learning step
        arma::vec nat_grad;
        int rnk = arma::rank(fisher);

        if (rnk == fisher.n_rows)
        {
            arma::mat H = arma::solve(fisher, gradient);
            nat_grad = H;
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;
            arma::mat H = arma::pinv(fisher);
            nat_grad = H * gradient;
        }
        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = nat_grad;
        //---

        arma::vec newvalues = policy.getParameters() + stepLength(gradient, nat_grad);
        policy.setParameters(newvalues);

        baseline_den.zeros();
        baseline_num.zeros();

        fisher.zeros();
    }

protected:
    std::vector<arma::vec> history_sumdlogpi;
    arma::vec sumdlogpi;
    arma::mat reward_EpStep;
    arma::cube sumGradLog_CompEpStep;
    arma::ivec maxsteps_Ep;

    unsigned int maxStepsPerEpisode, stepCount;

    arma::mat baseline_num, baseline_den, fisher, fisherEp;
};



///////////////////////////////////////////////////////////////////////////////////////
/// NATURAL REINFORCE ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////
template<class ActionC, class StateC>
class NaturalREINFORCEAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{

    USE_PGA_MEMBERS

public:
    NaturalREINFORCEAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                              unsigned int nbEpisodes, GradientStep& stepL,
                              bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj)
    {
        stepCount = 0;
    }

    NaturalREINFORCEAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                              unsigned int nbEpisodes, GradientStep& stepL,
                              RewardTransformation& reward_tr,
                              bool baseline = true) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, reward_tr, baseline)
    {
        stepCount = 0;
    }


    virtual ~NaturalREINFORCEAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init() override
    {
        unsigned int dp = policy.getParametersSize();
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(policy.getParametersSize()));

        baseline_den.zeros(dp);
        baseline_num.zeros(dp);

        sumdlogpi.set_size(dp);
        fisher.zeros(dp,dp);
        fisherEp.zeros(dp,dp);
    }

    virtual void initializeVariables() override
    {
        sumdlogpi.zeros();
        fisherEp.zeros();
        stepCount = 0;
    }

    virtual void updateStep(const Reward& reward) override
    {
        arma::vec grad = policy.difflog(currentState, currentAction);
        sumdlogpi += grad;

        fisherEp += grad * grad.t();
        stepCount++;
    }

    virtual void updateAtEpisodeEnd() override
    {
        for (int p = 0; p < baseline_num.n_elem; ++p)
        {
            baseline_num[p] += Jep * sumdlogpi[p] * sumdlogpi[p];
            baseline_den[p] += sumdlogpi[p] * sumdlogpi[p];
            history_sumdlogpi[epiCount][p] = sumdlogpi[p];
        }
        fisherEp /= stepCount;
        fisher += fisherEp;
    }

    virtual void updatePolicy() override
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
        if (task.gamma == 1.0)
            gradient /= totstep;
        else
            gradient /= nbEpisodesToEvalPolicy;
        fisher /= nbEpisodesToEvalPolicy;

        //--- Compute learning step

        arma::vec nat_grad;
        int rnk = arma::rank(fisher);
        //        std::cout << rnk << " " << fisher << std::endl;
        if (rnk == fisher.n_rows)
        {
            arma::mat H = arma::solve(fisher, gradient);
            nat_grad = H;
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;
            arma::mat H = arma::pinv(fisher);
            nat_grad = H * gradient;
        }
        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = nat_grad;
        //---

        arma::vec newvalues = policy.getParameters() + stepLength(gradient, nat_grad);
        policy.setParameters(newvalues);

        baseline_den.zeros();
        baseline_num.zeros();
        fisher.zeros();
    }

protected:

    arma::vec sumdlogpi, baseline_den, baseline_num;
    std::vector<arma::vec> history_sumdlogpi;
    arma::mat fisher, fisherEp;
    unsigned int stepCount;
};


}// end namespace ReLe

#endif //NATURALPOLICYGRADIENTALGORITHM_H_
