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

#ifndef INCLUDE_RELE_UTILS_LOGGER_TRANSITION_H_
#define INCLUDE_RELE_UTILS_LOGGER_TRANSITION_H_

namespace ReLe
{
template<class ActionC, class StateC>
struct Transition
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

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

}

#endif /* INCLUDE_RELE_UTILS_LOGGER_TRANSITION_H_ */
