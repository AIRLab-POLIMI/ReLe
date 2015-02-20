/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef SAMPLE_H_
#define SAMPLE_H_

#include "BasicsTraits.h"

namespace ReLe
{

template<class ActionC, class StateC>
struct Sample
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

    Sample(size_t x, unsigned int u, size_t xn, double r) :
        x(x), u(u), xn(xn), r(r)
    {

    }

    typename state_type<StateC>::type x;
    typename action_type<ActionC>::type u;
    typename state_type<StateC>::type xn;
    double r;
};

}

#endif /* SAMPLE_H_ */