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
#include "Sample.h"

#include <vector>
#include <iostream>

namespace ReLe
{

template<class ActionC, class StateC>
class Logger
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

    struct Transition
    {
        StateC x;
        ActionC u;
        StateC xn;
        Reward r;

        void init(const StateC& x)
        {
            this->x = x;
        }

        void update(const ActionC& u, const StateC& xn, const Reward& r)
        {
            this->u = u;
            this->xn = xn;
            this->r = r;
        }
    };

public:
    Logger(bool logTransitions) :
        logTransitions(logTransitions)
    {

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
        printTransitions();

        std::cout << std::endl << std::endl << "--- statistics ---" << std::endl
                  << std::endl;

        //print initial state
        std::cout << "- Initial State" << std::endl << "x(t = 0): "
                  << samples[0].x << std::endl;

        //printStateStatistics();

    }

private:
    void printTransitions()
    {
        if (logTransitions)
        {
            std::cout << "- Transitions" << std::endl;
            int t = 0;
            for (auto sample : samples)
            {
                auto& x = sample.x;
                auto& u = sample.u;
                auto& xn = sample.xn;
                Reward& r = sample.r;
                std::cout << "t = " << t++ << ": (" << x << " " << u
                          << " " << xn << " " << r << ")"
                          << std::endl;
            }
        }
    }

    /*template<class U>
    void printStateStatistics()
    {
    }

    template<>
    void printStateStatistics<FiniteState>()
    {
        std::cout << "- State Visits" << std::endl;
        std::size_t totalVisits = samples.size();
        std::size_t countedVisits = 0;
        for (std::size_t i = 0; countedVisits < totalVisits; i++)
        {
        	auto func = [] (const Transition& t) { return t.x.getStateN() == i; };
            std::size_t visits = std::count_if(samples.begin(), samples.end(),
                                               i);
            std::cout << "x(" << i << ") = " << visits << std::endl;
            countedVisits += visits;
        }
    }*/

private:
    bool logTransitions;
    Transition transition;
    std::vector<Transition> samples;

};

}

#endif /* INCLUDE_UTILS_LOGGER_H_ */
