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

#include <vector>
#include <fstream>

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

template <class ActionC, class StateC>
using Episode = std::vector< Transition<ActionC,StateC> >;

template<class ActionC, class StateC>
class TrajectoryData : public std::vector< Episode<ActionC,StateC> >
{
public:
    void WriteToStream(std::ostream& out)
    {
        int i, nbep = this->size();

        if (nbep > 0)
        {

            Transition<ActionC, StateC>& sample = (*this)[0][0];
            out << sample.x.serializedSize()  << ","
                << sample.u.serializedSize()  << ","
                << sample.r.size()  << std::endl;

            for (i = 0; i < nbep; ++i)
            {
                Episode<ActionC,StateC>& samples = this->at(i);
                size_t total = samples.size();
                size_t index = 0;
                for(auto& sample : samples)
                {
                    index++;
                    out << sample.x  << ","
                        << sample.u  << ","
                        << sample.xn << ","
                        << sample.r  << ","
                        << sample.xn.isAbsorbing() << ","
                        << (index == total) << std::endl;
                }
            }
        }
    }
};


}

#endif /* INCLUDE_RELE_UTILS_LOGGER_TRANSITION_H_ */
