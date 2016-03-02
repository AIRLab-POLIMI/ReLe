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

#ifndef INCLUDE_RELE_UTILS_LOGGER_STATESTATISTICGENERATOR_H_
#define INCLUDE_RELE_UTILS_LOGGER_STATESTATISTICGENERATOR_H_

#include <map>
#include <sstream>

namespace ReLe
{

template<class StateC>
class StateStatisticGenerator
{
public:
    void addStateVisit(StateC& state)
    {

    }

    std::string to_str()
    {
        std::stringstream ss;;
        ss << "No statistics for this type of state available" << std::endl;
        return ss.str();
    }
};


template<>
class StateStatisticGenerator<FiniteState>
{
public:
    StateStatisticGenerator()
    {
        visitsSize = 0;
    }

    void addStateVisit(FiniteState& state)
    {
        visits[state.getStateN()]++;
        visitsSize++;
    }

    std::string to_str()
    {
        std::stringstream ss;

        for(auto& pair : visits)
        {
            unsigned int stateN = pair.first;
            double visitsN = pair.second;
            ss << stateN << ": " << visitsN / visitsSize << std::endl;
        }

        return ss.str();
    }

private:
    std::map<unsigned int, int> visits;
    int visitsSize;
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGER_STATESTATISTICGENERATOR_H_ */
