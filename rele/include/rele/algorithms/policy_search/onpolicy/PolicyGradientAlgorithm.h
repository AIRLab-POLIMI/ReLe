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
#include "RewardTransformation.h"
#include <cassert>
#include <iomanip>

namespace ReLe
{

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
        runCount(0), epiCount(0), df(1.0), Jep(0.0),
        rewardTr(new IndexRT(reward_obj)), cleanRT(true),
        useBaseline(baseline), output2LogReady(false), stepLength(stepL),
        currentItStats(nullptr)
    {
    }

    AbstractPolicyGradientAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                                    unsigned int nbEpisodes, StepRule& stepL,
                                    RewardTransformation& reward_tr,
                                    bool baseline = true) :
        policy(policy), nbEpisodesToEvalPolicy(nbEpisodes),
        runCount(0), epiCount(0), df(1.0), Jep(0.0),
        rewardTr(&reward_tr), cleanRT(false),
        useBaseline(baseline), output2LogReady(false), stepLength(stepL),
        currentItStats(nullptr)
    {
    }

    virtual ~AbstractPolicyGradientAlgorithm()
    {
        if (cleanRT)
        {
            delete rewardTr;
        }
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
        typename action_type<ActionC>::type_ref u = action;
        u = policy(state);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      ActionC& action)
    {

        updateStep(reward);

        //calculate current J value
        Jep += df * rewardTr->operator ()(reward);
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
        Jep += df * rewardTr->operator ()(reward);

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
    RewardTransformation* rewardTr;
    bool cleanRT;

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
    using Base::rewardTr;                                          \
    using Base::history_J;                                         \
    using Base::useBaseline;                                       \
    using Base::output2LogReady;                                   \
    using Base::currentItStats;                                    \
    using Base::currentAction;                                     \
    using Base::currentState;


}// end namespace ReLe

#endif //POLICYGRADIENTALGORITHM_H_
