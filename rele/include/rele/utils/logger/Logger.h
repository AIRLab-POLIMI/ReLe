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

#include "Basics.h"
#include "Transition.h"
#include "LoggerStrategy.h"


#include <vector>
#include <iostream>
#include <memory>

namespace ReLe
{

template<class ActionC, class StateC>
class Logger
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    Logger()
    {
        strategy = nullptr;
    }

    void log(StateC& x)
    {
        transition.init(x);
    }

    void log(ActionC& u, StateC& xn, Reward& r)
    {
        transition.update(u, xn, r);
        samples.push_back(transition);
        transition.init(xn);
    }

    void printStatistics()
    {
        if(!strategy)
        {
            PrintStrategy<ActionC, StateC> strategy;
            strategy.processData(samples);
        }
        else
        {
            strategy->processData(samples);
        }
    }

    void setStrategy(LoggerStrategy<ActionC, StateC>* strategy)
    {
        this->strategy = strategy;
    }


private:
    Transition<ActionC, StateC> transition;
    std::vector<Transition<ActionC, StateC>> samples;

    LoggerStrategy<ActionC, StateC>* strategy;

};

}

#endif /* INCLUDE_UTILS_LOGGER_H_ */
