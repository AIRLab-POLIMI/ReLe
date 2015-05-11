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

#ifndef INCLUDE_RELE_CORE_ACTIONMASK_H_
#define INCLUDE_RELE_CORE_ACTIONMASK_H_

#include <vector>

namespace ReLe
{

template<class StateC, class OutputC>
class ActionMask
{
public:
    ActionMask(unsigned int size) : size(size)
    {

    }

    virtual std::vector<OutputC> getMask(typename state_type<StateC>::const_type_ref state) = 0;

    virtual ~ActionMask()
    {

    }

protected:
    unsigned int size;
};

}


#endif /* INCLUDE_RELE_CORE_ACTIONMASK_H_ */
