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

#ifndef INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_
#define INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_

#include <iostream>
#include "StateStatisticGenerator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class LoggerStrategy
{
public:
    virtual void processData(
        std::vector<Transition<ActionC, StateC>>& samples) = 0;
    virtual ~LoggerStrategy()
    {
    }

};

template<class ActionC, class StateC>
class PrintStrategy
{
public:
    PrintStrategy(bool logTransitions = true) :
        logTransitions(logTransitions)
    {

    }

    void processData(std::vector<Transition<ActionC, StateC>>& samples)
    {
        printTransitions(samples);

        std::cout << std::endl << std::endl << "--- statistics ---" << std::endl
                  << std::endl;

        //print initial state
        std::cout << "- Initial State" << std::endl << "x(t0) = ["
                  << samples[0].x << "]" << std::endl;

        printStateStatistics(samples);
    }

private:
    void printTransitions(std::vector<Transition<ActionC, StateC>>& samples)
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
                std::cout << "t = " << t++ << ": x = [" << x << "] u = [" << u
                          << "] xn = [" << xn << "] r = [" << r << "]"
                          << std::endl;
            }
        }
    }

    void printStateStatistics(std::vector<Transition<ActionC, StateC>>& samples)
    {
        std::cout << "- State Statistics" << std::endl;

        for(auto& transition : samples)
        {
            stateStatisticsGenerator.addStateVisit(transition.xn);
        }

        std::cout << stateStatisticsGenerator.to_str() << std::endl;

    }

private:
    bool logTransitions;
    StateStatisticGenerator<StateC> stateStatisticsGenerator;

};

template<class ActionC, class StateC>
class WriteStrategy
{
public:
    WriteStrategy(const std::string& path) :
        path(path)
    {

    }

    void processData(std::vector<Transition<ActionC, StateC>>& samples)
    {
        std::ofstream ofs(path); //TODO append?

        //TODO print data as matrix


        ofs.close();
    }

private:
    const std::string& path;
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_ */
