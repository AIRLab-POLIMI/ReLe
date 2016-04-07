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

/*
 * Written by: Carlo D'Eramo
 */

#include "rele/policy/nonparametric/SequentialPolicy.h"


namespace ReLe
{

SequentialPolicy::SequentialPolicy(unsigned int nActions, unsigned int episodeLength) :
    currentAction(0),
    currentEpisodeStep(0),
    episodeLength(episodeLength)
{
    this->nactions = nActions;
}

unsigned int SequentialPolicy::operator()(const size_t& state)
{
    unsigned int returnValue = currentAction;
    currentAction++;

    if(currentEpisodeStep == episodeLength)
    {
        currentAction--;
        currentEpisodeStep = -1;
    }
    else if(currentAction == nactions)
        currentAction = 0;

    currentEpisodeStep++;

    return returnValue;
}

double SequentialPolicy::operator()(const size_t& state, const unsigned int& action)
{
    return action == currentAction;
}

inline std::string SequentialPolicy::getPolicyName()
{
    return "Sequential";
}

SequentialPolicy* SequentialPolicy::clone()
{
    return new SequentialPolicy(*this);
}

}
