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

/*!
 * A Terminal condition can be used to signal to the core that the agent estimate as converged.
 * This abstract class can be overloaded and used by the agents to stop the experiments when
 * enough data has been collected.
 */
class TerminalCondition
{
public:
    virtual ~TerminalCondition()
    {
    }

    virtual bool checkCond() = 0;
};

/*!
 * The Agent is the basic interface of all online agents.
 * All online algorithms should extend this abstract class.
 * The Agent interface provides all the methods that can be used to interact with an MDP through
 * the Core class. It includes methods to run the learning over an MDP and to test the learned policy.
 */
template<class ActionC, class StateC>
class Agent
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");
public:
    Agent() : terminalCond(nullptr)
    {
    }

    /*!
     * This method is called at the beginning of each test episode. by default does nothing,
     * but can be overloaded
     */
    virtual void initTestEpisode() {}

    /*!
     * This method is called at the beginning of each learning episode. Must be implemented.
     * Normally this method contains the algorithm initialization.
     * \param state the initial MDP state
     * \param action the action selected by the agent in the initial state
     */
    virtual void initEpisode(const StateC& state, ActionC& action) = 0;

    /*!
     * This method is used to sample an action in test episodes. Must be implemented.
     * Normally, this method is trivial, as it just sample an action from a policy.
     * \param state the current MDP state
     * \param action the action selected by the agent in the current state
     */
    virtual void sampleAction(const StateC& state, ActionC& action) = 0;

    /*!
     * This method is used during each learning step. Must be implemented.
     * Normally this method contains the learning algorithm, for step-based agents, or data collection
     * for episode-based agents.
     * \param reward the reward achieved in the previous learning step.
     * \param nextState the state reached after the previous learning state i.e. the current state
     * \param action the action selected by the agent in the current state
     */
    virtual void step(const Reward& reward, const StateC& nextState,
                      ActionC& action) = 0;

    /*!
     * This method is called if an episode ends in a terminal state. Must be implemented.
     * Normally this method contains the learning algorithm.
     * \param reward the reward achieved after reaching the terminal state.
     */
    virtual void endEpisode(const Reward& reward) = 0;

    /*!
     * This method is called if an episode ends after reaching the maximum number of iterations.
     * Must be implemented.
     * Normally this method contains the learning algorithm.
     */
    virtual void endEpisode() = 0;

    /*!
     * This method is used to log agent step informations.
     * Can be overloaded to return information that can be processed by the logger.
     * By default a null pointer is returned, which means that no data will be logged.
     * \return the data to be logged from the agent at the current step.
     */
    virtual AgentOutputData* getAgentOutputData()
    {
        return nullptr;
    }

    /*!
     * This method is used to log agent informations at episode end.
     * Can be overloaded to return information that can be processed by the logger.
     * By default a null pointer is returned, which means that no data will be logged.
     * \return the data to be logged from the agent at the episode end.
     */
    virtual AgentOutputData* getAgentOutputDataEnd()
    {
        return nullptr;
    }

    /*!
     * This method is called before each learning step and return if the terminal condition
     * has been reached. Terminal condition can be implemented by setting the Agent::terminalCond member.
     * \return whether the terminal condition has been reached.
     */
    inline bool isTerminalConditionReached()
    {
        if (terminalCond == nullptr)
            return false;
        else
            return terminalCond->checkCond();
    }

    /*!
     * This method sets the agent task, i.e. the environment properties. This method also calls Agent::init()
     * \param task the task properties of the environment
     */
    void setTask(const EnvironmentSettings& task)
    {
        this->task = task;
        this->init();
    }


    virtual ~Agent()
    {

    }

protected:
    /*!
     * This method is called after the agent task has been set.
     * By default does nothing, but can be overloaded with agent initialization, e.g. Q table allocation.
     */
    virtual void init()
    {

    }


protected:
    //! The task that the agent will perform
    EnvironmentSettings task;
    //! The terminal condition of the agent
    TerminalCondition* terminalCond;
};

}

#endif /* AGENT_H_ */
