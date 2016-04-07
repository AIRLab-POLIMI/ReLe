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

#include "rele/policy/nonparametric/DeterministicPolicy.h"

#include <sstream>

namespace ReLe
{

unsigned int DeterministicPolicy::operator()(const size_t& state)
{
    return pi(state);
}

double DeterministicPolicy::operator()(const size_t& state, const unsigned int& action)
{
    if(action == pi(state))
        return 1.0;
    else
        return 0.0;
}

std::string DeterministicPolicy::printPolicy()
{
    //TODO [MINOR] choose policy format
    std::stringstream ss;
    ss << "- Policy" << std::endl;
    for (unsigned int i = 0; i < pi.n_elem; i++)
    {
        ss << "policy(" << i << ") = " << pi(i) << std::endl;
    }

    return ss.str();
}


}
