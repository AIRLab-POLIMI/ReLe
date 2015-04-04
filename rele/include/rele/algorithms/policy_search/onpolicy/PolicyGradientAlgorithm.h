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

#ifndef POLICYGRADIENTALGORITHM_H_
#define POLICYGRADIENTALGORITHM_H_

#include "Agent.h"
#include "Policy.h"
#include "Basics.h"
#include "BasicFunctions.h"
#include "policy_search/onpolicy/GradientOutputData.h"
#include "policy_search/step_rules/StepRules.h"
#include <cassert>
#include <iomanip>

namespace ReLe
{

////Templates needed to sample different action types
template<class StateC, class PolicyC>
arma::vec diffLogWorker(const StateC& state, FiniteAction& action, PolicyC& policy)
{
    unsigned int u = action.getActionN();
    return policy.difflog(state, u);
}

template<class StateC, class ActionC, class PolicyC>
arma::vec diffLogWorker(const StateC& state, ActionC& action, PolicyC& policy)
{
    return policy.difflog(state, action);
}


///////////////////////////////////////////////////////////////////////////////////////
/// ABSTRACT GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////
template<class ActionC, class StateC>
class AbstractPolicyGradientAlgorithm: public Agent<ActionC, StateC>
{

public:
    AbstractPolicyGradientAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                                    unsigned int nbEpisodes, StepRule& stepL,
                                    bool baseline = true, int reward_obj = 0) :
        policy(policy), nbEpisodesToEvalPolicy(nbEpisodes),
        runCount(0), epiCount(0), df(1.0), Jep(0.0), rewardId(reward_obj),
        useBaseline(baseline), output2LogReady(false), stepLength(stepL),
        currentItStats(nullptr)
    {
    }

    virtual ~AbstractPolicyGradientAlgorithm()
    {
    }

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, ActionC& action)
    {
        df = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode

        // Initialize variables
        initializeVariables();

        //--- set up agent output
        if (epiCount == 0)
        {
            currentItStats = new GradientIndividual();
            currentItStats->policy_parameters = policy.getParameters();
        }
        //---

        sampleAction(state, action);

        // save state and action for late use
        currentState  = state;
        currentAction = action;
    }

    virtual void initTestEpisode()
    {
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
    {
        sampleActionWorker(state, action, policy);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      ActionC& action)
    {

        updateStep(reward);

        //calculate current J value
        Jep += df * reward[rewardId];
        //update discount factor
        df *= this->task.gamma;

        sampleAction(nextState, action);

        // save state and action for late use
        currentState  = nextState;
        currentAction = action;
    }

    virtual void endEpisode(const Reward& reward)
    {
        updateStep(reward);

        //add last contribute
        Jep += df * reward[rewardId];

        //perform remaining operation
        this->endEpisode();

    }

    virtual void endEpisode()
    {

        //save policy value
        history_J[epiCount] = Jep;

        updateAtEpisodeEnd();

        //last episode is the number epiCount+1
        epiCount++;
        //check evaluation of actual policy
        if (epiCount == nbEpisodesToEvalPolicy)
        {
            updatePolicy();

            //reset counters and gradient
            epiCount = 0; //reset episode counter
            runCount++; //update run counter
            output2LogReady = true; //output must be ready for log
        }
    }

    virtual AgentOutputData* getAgentOutputDataEnd()
    {
        if (output2LogReady)
        {
            //output is ready, activate flag
            output2LogReady = false;
            return currentItStats;
        }
        return nullptr;
    }

protected:
    virtual void init()
    {
        history_J.assign(nbEpisodesToEvalPolicy, 0.0);
    }

    virtual void initializeVariables() = 0;
    virtual void updateStep(const Reward& reward) = 0;
    virtual void updateAtEpisodeEnd() = 0;
    virtual void updatePolicy() = 0;

protected:
    DifferentiablePolicy<ActionC, StateC>& policy;
    unsigned int nbEpisodesToEvalPolicy;
    unsigned int runCount, epiCount;
    double df, Jep;
    StepRule& stepLength;
    int rewardId;

    std::vector<double> history_J;
    bool useBaseline, output2LogReady;
    GradientIndividual* currentItStats;
    ActionC currentAction;
    StateC currentState;
};

#define USE_PGA_MEMBERS                                            \
    typedef AbstractPolicyGradientAlgorithm<ActionC, StateC> Base; \
    using Base::policy;                                            \
    using Base::nbEpisodesToEvalPolicy;                            \
    using Base::runCount;                                          \
    using Base::epiCount;                                          \
    using Base::df;                                                \
    using Base::Jep;                                               \
    using Base::stepLength;                                        \
    using Base::rewardId;                                          \
    using Base::history_J;                                         \
    using Base::useBaseline;                                       \
    using Base::output2LogReady;                                   \
    using Base::currentItStats;                                    \
    using Base::currentAction;                                     \
    using Base::currentState;


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
        arma::vec grad = diffLogWorker(currentState, currentAction, policy);
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
        arma::mat eMetric = arma::eye(nbParams,nbParams);
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


///////////////////////////////////////////////////////////////////////////////////////
/// GPOMDP GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////

template<class ActionC, class StateC>
class GPOMDPAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{
    USE_PGA_MEMBERS


public:
    enum BaseLineType { MULTI, SINGLE };


    GPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                    unsigned int nbEpisodes, unsigned int nbSteps, StepRule& stepL,
                    BaseLineType btype, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, true, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(btype)
    {
    }

    GPOMDPAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                    unsigned int nbEpisodes, unsigned int nbSteps, StepRule& stepL,
                    int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, false, reward_obj),
        maxStepsPerEpisode(nbSteps),
        bType(BaseLineType::SINGLE)
    {
    }

    virtual ~GPOMDPAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init()
    {
        unsigned int dp = policy.getParametersSize();
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
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

        int dp = policy.getParametersSize();

        arma::vec grad = diffLogWorker(currentState, currentAction, policy);
        sumdlogpi += grad;

        // store the basic elements used to compute the gradient
        reward_EpStep(epiCount, stepCount) = df * reward[rewardId];

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
                baseline_num(p,stepCount) += df * reward[rewardId] * sumdlogpi(p) * sumdlogpi(p);
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
                baseline_num1_single(p) += df * reward[rewardId] * sumdlogpi(p);
                baseline_num2_single(p) += sumdlogpi(p);
            }
        }

        stepCount++;
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
    }

    virtual void updatePolicy()
    {
        int nbParams = policy.getParametersSize();
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

        arma::vec newvalues = policy.getParameters() + gradient * step_size;
        policy.setParameters(newvalues);
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
                           unsigned int nbEpisodes, unsigned int nbSteps, StepRule& stepL,
                           bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj),
        maxStepsPerEpisode(nbSteps)
    {
    }

    virtual ~NaturalGPOMDPAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init()
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

    virtual void initializeVariables()
    {
        sumdlogpi.zeros();
        stepCount = 0;
        fisherEp.zeros();
    }

    virtual void updateStep(const Reward& reward)
    {

        int dp = policy.getParametersSize();

        arma::vec grad = diffLogWorker(currentState, currentAction, policy);
        sumdlogpi += grad;

        fisherEp += grad * grad.t();

        // store the basic elements used to compute the gradient
        reward_EpStep(epiCount, stepCount) = df * reward[rewardId];

        for (int p = 0; p < dp; ++p)
        {
            sumGradLog_CompEpStep(p,epiCount,stepCount) = sumdlogpi(p);
        }

        // compute the baseline

        for (int p = 0; p < dp; ++p)
        {
            baseline_num(p,stepCount) += df * reward[rewardId] * sumdlogpi(p) * sumdlogpi(p);
        }

        for (int p = 0; p < dp; ++p)
        {
            baseline_den(p,stepCount) += sumdlogpi(p) * sumdlogpi(p);
        }

        stepCount++;
    }

    virtual void updateAtEpisodeEnd()
    {
        maxsteps_Ep(epiCount) = stepCount;
        fisherEp /= stepCount;
        fisher += fisherEp;
    }

    virtual void updatePolicy()
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
        gradient /= nbEpisodesToEvalPolicy;
        fisher /= nbEpisodesToEvalPolicy;

        //--- Compute learning step

        arma::vec step_size = stepLength.stepLength(gradient, fisher);

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
        currentItStats->stepLength = step_size;
        //---

        //        std::cout << step_length << std::endl;
        //        std::cout << nat_grad.t();

        arma::vec newvalues = policy.getParameters() + step_size * nat_grad;
        policy.setParameters(newvalues);
        //        std::cout << "new_params: "  << newvalues.t();

        for (int p = 0; p < nbParams; ++p)
        {
            for (int t = 0; t < maxStepsPerEpisode; ++t)
            {
                baseline_den(p,t) = 0;
                baseline_num(p,t) = 0;
            }
        }
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
                              unsigned int nbEpisodes, StepRule& stepL,
                              bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj)
    {
    }

    virtual ~NaturalREINFORCEAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init()
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

    virtual void initializeVariables()
    {
        sumdlogpi.zeros();
        fisherEp.zeros();
        stepCount = 0;
    }

    virtual void updateStep(const Reward& reward)
    {
        arma::vec grad = diffLogWorker(currentState, currentAction, policy);
        sumdlogpi += grad;

        fisherEp += grad * grad.t();
        stepCount++;
    }

    virtual void updateAtEpisodeEnd()
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
        fisher /= nbEpisodesToEvalPolicy;

        //--- Compute learning step

        arma::vec step_size = stepLength.stepLength(gradient, fisher);

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
        currentItStats->stepLength = step_size;
        //---

        //        std::cout << step_length << std::endl;
        //        std::cout << nat_grad.t();

        arma::vec newvalues = policy.getParameters() + step_size * nat_grad;
        policy.setParameters(newvalues);
        //        std::cout << "new_params: "  << newvalues.t();

        for (int i = 0; i < nbParams; ++i)
        {
            baseline_den[i] = 0;
            baseline_num[i] = 0;
        }
        fisher.zeros();
    }

protected:

    arma::vec sumdlogpi, baseline_den, baseline_num;
    std::vector<arma::vec> history_sumdlogpi;
    arma::mat fisher, fisherEp;
    unsigned int stepCount;
};



///////////////////////////////////////////////////////////////////////////////////////
/// eNAC GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////
/**
 * Policy Gradient Methods for Robotics
 * Jan Peters, Stefan Schaal
 * IROS 2006
 * http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/IROS2006-Peters_%5b0%5d.pdf
 */

#define AUGMENTED

template<class ActionC, class StateC>
class eNACAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{
    USE_PGA_MEMBERS

public:
    eNACAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                  unsigned int nbEpisodes, StepRule& stepL,
                  bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj)
    {
    }

    virtual ~eNACAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init()
    {
        unsigned int dp = policy.getParametersSize();
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(dp));
#ifdef AUGMENTED
        fisher.zeros(dp+1,dp+1);
        g.zeros(dp+1);
        eligibility.zeros(dp+1);
        phi.zeros(dp+1);
#else
        fisher.zeros(dp,dp);
        g.zeros(dp);
        eligibility.zeros(dp);
        phi.zeros(dp);
#endif
        Jpol = 0.0;
    }

    virtual void initializeVariables()
    {
        phi.zeros();
#ifdef AUGMENTED
        phi(policy.getParametersSize()) = 1.0;
#endif
    }

    virtual void updateStep(const Reward& reward)
    {
        // get number of parameters
        int dp = policy.getParametersSize();

        // Compute the derivative of the logarithm of the policy and
        // Evaluate it in (s_t, a_t)
        arma::vec grad = diffLogWorker(currentState, currentAction, policy);

        //Construct basis functions
        for (unsigned int i = 0; i < dp; ++i)
            phi[i] += df * grad[i];
    }

    virtual void updateAtEpisodeEnd()
    {
        Jpol += Jep;
        fisher += phi * phi.t();
        g += Jep * phi;
        eligibility += phi;
    }

    virtual void updatePolicy()
    {
        // get number of parameters
        int dp = policy.getParametersSize();

        // compute mean value
        fisher /= nbEpisodesToEvalPolicy;
        g /= nbEpisodesToEvalPolicy;
        eligibility /= nbEpisodesToEvalPolicy;
        Jpol /= nbEpisodesToEvalPolicy;
        int nbParams = policy.getParametersSize();

        //--- Compute learning step

        arma::vec step_size;
        arma::vec nat_grad;
        int rnk = arma::rank(fisher);
        //        std::cout << rnk << " " << fisher << std::endl;
        if (rnk == fisher.n_rows)
        {
            arma::vec grad;
            if (useBaseline == true)
            {
                arma::mat tmp = arma::solve(nbEpisodesToEvalPolicy * fisher - eligibility * eligibility.t(), eligibility);
                arma::mat Q = (1 + eligibility.t() * tmp) / nbEpisodesToEvalPolicy;
                arma::mat b = Q * (Jpol - eligibility.t() * arma::solve(fisher, g));
                grad = g - eligibility * b;
                nat_grad = arma::solve(fisher, grad);
            }
            else
            {
                grad = g;
                nat_grad = arma::solve(fisher, grad);
            }

            step_size = stepLength.stepLength(grad, fisher);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            arma::vec grad;
            if (useBaseline == true)
            {
                arma::mat b = (1 + eligibility.t() * arma::pinv(nbEpisodesToEvalPolicy * fisher - eligibility * eligibility.t()) * eligibility)
                              * (Jpol - eligibility.t() * H * g)/ nbEpisodesToEvalPolicy;
                grad = g - eligibility * b;
                nat_grad = H * (grad);
            }
            else
            {
                grad = g;
                nat_grad = H * grad;
            }
            step_size = stepLength.stepLength(grad, fisher);
        }
        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = nat_grad.rows(0,dp-1);
        currentItStats->stepLength = step_size;
        //---

        //        std::cout << stepLength <<std::endl;
        //        std::cout << nat_grad.t();

        arma::vec newvalues = policy.getParameters() + step_size * nat_grad.rows(0,dp-1);
        policy.setParameters(newvalues);
        //        std::cout << "new_params: "  << newvalues.t();

        for (int p = 0; p < nbParams; ++p)
        {
            eligibility[p] = 0.0;
            g[p] = 0.0;
        }
        fisher.zeros();
        Jpol = 0.0;
    }

protected:
    std::vector<arma::vec> history_sumdlogpi;
    arma::vec g, eligibility, phi;

    arma::mat fisher;
    double Jpol;
};

}// end namespace ReLe

#endif //POLICYGRADIENTALGORITHM_H_
