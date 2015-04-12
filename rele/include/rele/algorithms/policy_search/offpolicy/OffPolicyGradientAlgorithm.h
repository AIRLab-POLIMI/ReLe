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

#ifndef OFFPOLICYGRADIENTALGORITHM_H_
#define OFFPOLICYGRADIENTALGORITHM_H_

#include "Agent.h"
#include "Distribution.h"
#include "Policy.h"
#include "Basics.h"
#include "BasicFunctions.h"
#include <cassert>
#include <iomanip>
#include "policy_search/offpolicy/OffGradientOutputData.h"

namespace ReLe
{

//Templates needed to handle different action types
template<class StateC, class PolicyC, class PolicyC2>
double PureOffPolicyGradientAlgorithmStepWorker(const StateC& state, const FiniteAction& action, PolicyC& policy, PolicyC2& behav,
        double& iwb, double& iwt, arma::vec& grad)
{
    typename action_type<FiniteAction>::type_ref u = action.getActionN();

    double valb = behav(state,u);
    double valt = policy(state,u);

    iwt *= valt;
    iwb *= valb;

    //init the sum of the gradient of the policy logarithm
    arma::vec logGradient = policy.difflog(state, u);
    grad += logGradient;

    return valt/valb;
}

template<class StateC, class ActionC, class PolicyC, class PolicyC2>
double PureOffPolicyGradientAlgorithmStepWorker(const StateC& state, const ActionC& action, PolicyC& policy, PolicyC2& behav,
        double& iwb, double& iwt, arma::vec& grad)
{
    double valb = behav(state,action);
    double valt = policy(state,action);

    iwt *= valt;
    iwb *= valb;

    //init the sum of the gradient of the policy logarithm
    arma::vec logGradient = policy.difflog(state, action);
    grad += logGradient;

    return valt/valb;
}


template<class ActionC, class StateC>
class PureOffPolicyGradientAlgorithm: public BatchAgent<ActionC, StateC>
{

public:
    PureOffPolicyGradientAlgorithm(DifferentiablePolicy<ActionC, StateC>& target_pol,
                                   Policy<ActionC, StateC>& behave_pol,
                                   unsigned int nbEpisodes, unsigned int nbSamplesForJ,
                                   double stepL = 0.5,
                                   bool baseline = true, int reward_obj = 0) :
        target(target_pol), behavioral(behave_pol), nbEpisodesperUpdate(nbEpisodes), runCounter(0),
        epCounter(0), df(1.0), Jep(0.0), rewardId(reward_obj),
        useBaseline(baseline), output2LogReady(false),
        currentItStats(nullptr), stepLength(stepL),
        nbIndipendentSamples(std::min(std::max(1,static_cast<int>(nbSamplesForJ)), static_cast<int>(nbEpisodes*0.1)))
    {
    }

    virtual ~PureOffPolicyGradientAlgorithm()
    {
    }

    inline void setStepLength(double penal)
    {
        stepLength = penal;
    }

    inline double getStepLength()
    {
        return stepLength;
    }

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, const ActionC& action)
    {
        df  = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode
        Jepoff = 0.0;

        //--- set up agent output
        if (epCounter == 0)
        {
            sumIWOverRun = 0.0;
            prodImpWeightT = 1.0;
            prodImpWeightB = 1.0;

            currentItStats = new OffGradientIndividual();
            currentItStats->policy_parameters = target.getParameters();
        }
        //---

        prodImpWeightB = 1.0;
        prodImpWeightT = 1.0;
        sumdlogpi.zeros(target.getParametersSize());
        currentState  = state;
        currentAction = action;

        //        std::cout << std::endl;
    }

    virtual void initTestEpisode()
    {
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
    {
        sampleActionWorker(state, action, target);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      const ActionC& nextAction)
    {


        //        std::cout << currentState << " " << currentAction << " " << reward[0] << std::endl;

        double currentIW = PureOffPolicyGradientAlgorithmStepWorker(currentState, currentAction, target, behavioral, prodImpWeightB, prodImpWeightT, sumdlogpi);

        //calculate current J value
        Jep += df * reward[rewardId];
        Jepoff += df * currentIW * reward[rewardId];
        //update discount factor
        df *= this->task.gamma;

        currentState  = nextState;
        currentAction = nextAction;
    }

    virtual void endEpisode(const Reward& reward)
    {


        //        std::cout << currentState << " " << currentAction << " " << reward[0] << std::endl;

        double currentIW = PureOffPolicyGradientAlgorithmStepWorker(currentState, currentAction, target, behavioral, prodImpWeightB, prodImpWeightT, sumdlogpi);

        //add last contribute
        Jep += df * reward[rewardId];
        Jepoff += df * currentIW * reward[rewardId];
        //perform remaining operation
        this->endEpisode();

    }

    virtual void endEpisode()
    {

        history_J[epCounter] = Jep;
        history_J_off[epCounter] = Jepoff;
        history_impWeights[epCounter] = prodImpWeightT / prodImpWeightB;
        history_sumdlogpi[epCounter] = sumdlogpi;
        sumIWOverRun += history_impWeights[epCounter];

        ++epCounter;

        if (epCounter == nbEpisodesperUpdate)
        {
            //all policies have been evaluated
            //conclude gradient estimate and update the distribution
            updatePolicy();

            //reset counters and gradient
            sumIWOverRun = 0.0;
            epCounter = 0; //reset policy counter
            runCounter++; //update run counter
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
    virtual void init() = 0;
    virtual void updatePolicy() = 0;


protected:
    DifferentiablePolicy<ActionC, StateC>& target;
    Policy<ActionC, StateC>& behavioral;
    unsigned int nbEpisodesperUpdate;
    unsigned int runCounter, epCounter;
    double df, Jep, Jepoff, stepLength, sumIWOverRun;
    int rewardId;

    double prodImpWeightB, prodImpWeightT;
    arma::vec sumdlogpi;

    std::vector<double> history_J;
    std::vector<double> history_J_off;
    std::vector<double> history_impWeights;
    std::vector<arma::vec> history_sumdlogpi;

    bool useBaseline, output2LogReady;
    OffGradientIndividual* currentItStats;

    unsigned int nbIndipendentSamples;
    ActionC currentAction;
    StateC currentState;
};

#define USE_PUREOFFPGA_MEMBERS                                               \
    typedef PureOffPolicyGradientAlgorithm<ActionC, StateC> Base;            \
    using Base::target;                                                      \
    using Base::behavioral;                                                  \
    using Base::nbEpisodesperUpdate;                                         \
    using Base::runCounter;                                                  \
    using Base::epCounter;                                                   \
    using Base::df;                                                          \
    using Base::Jep;                                                         \
    using Base::Jepoff;                                                      \
    using Base::stepLength;                                                  \
    using Base::sumIWOverRun;                                                \
    using Base::rewardId;                                                    \
    using Base::history_J;                                                   \
    using Base::history_J_off;                                               \
    using Base::history_impWeights;                                          \
    using Base::history_sumdlogpi;                                           \
    using Base::useBaseline;                                                 \
    using Base::output2LogReady;                                             \
    using Base::currentItStats;                                              \
    using Base::nbIndipendentSamples;                                        \
    using Base::currentAction;                                               \
    using Base::currentState;

} //end namespace

#endif //OFFPOLICYGRADIENTALGORITHM_H_
