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
#include "policy_search/gradient/offpolicy/OffGradientOutputData.h"
#include "policy_search/step_rules/StepRules.h"

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

template<class ActionC, class StateC>
class AbstractOffPolicyGradientAlgorithm: public BatchAgent<ActionC, StateC>
{

public:
    AbstractOffPolicyGradientAlgorithm(DifferentiablePolicy<ActionC, StateC>& target_pol,
                                       Policy<ActionC, StateC>& behave_pol,
                                       unsigned int nbEpisodes, unsigned int nbSamplesForJ,
                                       StepRule& stepL, bool baseline = true, int reward_obj = 0) :
        target(target_pol), behavioral(behave_pol), nbEpisodesperUpdate(nbEpisodes), runCounter(0),
        epCounter(0), df(1.0), Jep(0.0), rewardId(reward_obj),
        useBaseline(baseline), output2LogReady(false),
        currentItStats(nullptr), stepRule(stepL),
        nbIndipendentSamples(std::min(std::max(1,static_cast<int>(nbSamplesForJ)), static_cast<int>(nbEpisodes*0.1)))
    {
        Jepoff = 0;
    }

    virtual ~AbstractOffPolicyGradientAlgorithm()
    {
    }

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, const ActionC& action)
    {
        df  = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode
        Jepoff = 0.0;

        // Initialize variables
        initializeVariables();

        //--- set up agent output
        if (epCounter == 0)
        {
            currentItStats = new OffGradientIndividual();
            currentItStats->policy_parameters = target.getParameters();
        }
        //---

        currentState  = state;
        currentAction = action;

        //        std::cout << std::endl;
    }

    virtual void initTestEpisode()
    {
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
    {
        typename action_type<ActionC>::type_ref u = action;
        u = target(state);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      const ActionC& nextAction)
    {
        double currentIW = updateStep(reward);

        updateStep(reward);

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
        double currentIW = updateStep(reward);

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

        updateAtEpisodeEnd();

        //last episode is the number epiCount+1
        ++epCounter;
        //check evaluation of actual policy
        if (epCounter == nbEpisodesperUpdate)
        {
            //all policies have been evaluated
            //conclude gradient estimate and update the distribution
            updatePolicy();

            //reset counters and gradient
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
    virtual void initializeVariables() = 0;
    virtual double updateStep(const Reward& reward) = 0;
    virtual void updateAtEpisodeEnd() = 0;
    virtual void updatePolicy() = 0;


protected:
    DifferentiablePolicy<ActionC, StateC>& target;
    Policy<ActionC, StateC>& behavioral;
    unsigned int nbEpisodesperUpdate;
    unsigned int runCounter, epCounter;
    double df, Jep, Jepoff;
    StepRule& stepRule;
    int rewardId;

    std::vector<double> history_J;
    std::vector<double> history_J_off;

    bool useBaseline, output2LogReady;
    OffGradientIndividual* currentItStats;

    unsigned int nbIndipendentSamples;
    ActionC currentAction;
    StateC currentState;
};

#define USE_PUREOFFPGA_MEMBERS                                               \
    typedef AbstractOffPolicyGradientAlgorithm<ActionC, StateC> Base;        \
    using Base::target;                                                      \
    using Base::behavioral;                                                  \
    using Base::nbEpisodesperUpdate;                                         \
    using Base::runCounter;                                                  \
    using Base::epCounter;                                                   \
    using Base::df;                                                          \
    using Base::Jep;                                                         \
    using Base::Jepoff;                                                      \
    using Base::stepRule;                                                    \
    using Base::rewardId;                                                    \
    using Base::history_J;                                                   \
    using Base::history_J_off;                                               \
    using Base::useBaseline;                                                 \
    using Base::output2LogReady;                                             \
    using Base::currentItStats;                                              \
    using Base::nbIndipendentSamples;                                        \
    using Base::currentAction;                                               \
    using Base::currentState;

} //end namespace

#endif //OFFPOLICYGRADIENTALGORITHM_H_
