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

#ifndef AGENT_H_
#define AGENT_H_

#include <vector>

#include "rele/core/Basics.h"

namespace ReLe
{

class TerminalCondition
{
public:
    virtual ~TerminalCondition()
    {
    }

    virtual bool checkCond() = 0;
};

template<class ActionC, class StateC>
class Agent
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");
public:
    Agent() : terminalCond(nullptr)
    {
    }

    virtual void initTestEpisode() {}
    virtual void initEpisode(const StateC& state, ActionC& action) = 0;
    virtual void sampleAction(const StateC& state, ActionC& action) = 0;
    virtual void step(const Reward& reward, const StateC& nextState,
                      ActionC& action) = 0;
    virtual void endEpisode(const Reward& reward) = 0;
    virtual void endEpisode() = 0;

    virtual AgentOutputData* getAgentOutputData()
    {
        return nullptr;
    }

    virtual AgentOutputData* getAgentOutputDataEnd()
    {
        return nullptr;
    }

    virtual inline bool isTerminalConditionReached()
    {
        if (terminalCond == nullptr)
            return false;
        else
            return terminalCond->checkCond();
    }

    void setTask(const EnvironmentSettings& task)
    {
        this->task = task;
        this->init();
    }


    virtual ~Agent()
    {

    }

protected:
    virtual void init()
    {

    }


protected:
    EnvironmentSettings task;
    TerminalCondition* terminalCond;
};


template<class ActionC, class StateC>
class BatchAgent : public Agent<ActionC, StateC>
{
public:

    void initEpisode(const StateC& state, ActionC& action) override
    {
        const ActionC& constref = action;
        this->initEpisode(state, constref);
    }

    void step(const Reward& reward, const StateC& nextState,
              ActionC& action) override
    {
        const ActionC& constref = action;
        this->step(reward, nextState, constref);
    }

    virtual void initEpisode(const StateC& state, const ActionC& action) = 0;
    virtual void step(const Reward& reward, const StateC& nextState,
                      const ActionC& action) = 0;

    virtual ~BatchAgent() {}

};

}

#endif /* AGENT_H_ */
