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

#ifndef INCLUDE_UTILS_LOGGER_H_
#define INCLUDE_UTILS_LOGGER_H_

#include "rele/core/Basics.h"
#include "rele/core/logger/LoggerStrategy.h"
#include "rele/core/Transition.h"

#include <vector>
#include <iostream>
#include <memory>

namespace ReLe
{

/*!
 * This class implement the logger for the ReLe::Core class.
 * this class is used to log both agent output data and environment trajectories into a dataset.
 * The behavior of this class depends on the logger strategy selected. If no strategy is specified
 * (or if the strategy is set to a null pointer) no logging will be performed.
 */
template<class ActionC, class StateC>
class Logger
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    /*!
     * Constructor.
     */
    Logger()
    {
        strategy = nullptr;
    }

    /*!
     * Method used to log the initial state
     * \param x the initial environment state
     */
    void log(StateC& x)
    {
        transition.init(x);
    }

    /*!
     * Method used to log a state transition
     * \param u the action performed by the agent
     * \param xn the state reached after the agent's action
     * \param r the reward achieved by the agent
     */
    void log(ActionC& u, StateC& xn, Reward& r)
    {
        transition.update(u, xn, r);
        episode.push_back(transition);
        transition.init(xn);
    }

    /*!
     * Method used to log agent output data
     * \param data the agent output data
     * \param step the current agent step
     */
    void log(AgentOutputData* data, unsigned int step)
    {
        if(data)
        {
            data->setStep(step);
            outputData.push_back(data);
        }
    }

    /*!
     * This method performs the logger operations
     */
    void printStatistics()
    {
        if(strategy)
        {
            strategy->processData(episode);
            strategy->processData(outputData);
        }
    }

    /*!
     * Setter.
     * \param strategy the selected logger strategy to be performed
     */
    void setStrategy(LoggerStrategy<ActionC, StateC>* strategy)
    {
        this->strategy = strategy;
    }


private:
    Transition<ActionC, StateC> transition;
    Episode<ActionC, StateC> episode;
    std::vector<AgentOutputData*> outputData;

    LoggerStrategy<ActionC, StateC>* strategy;

};

}

#endif /* INCLUDE_UTILS_LOGGER_H_ */
