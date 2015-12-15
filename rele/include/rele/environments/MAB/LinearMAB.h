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

#ifndef INCLUDE_RELE_ENVIRONMENTS_MAB_LINEARMAB_H_
#define INCLUDE_RELE_ENVIRONMENTS_MAB_LINEARMAB_H_

#include "MAB/MAB.h"

namespace ReLe
{

template<class ActionC>
class LinearMAB: public MAB<ActionC>
{
public:
    LinearMAB(double gamma, unsigned int horizon = 1) : MAB<ActionC>(gamma, horizon)
    {
    }

    virtual double computeReward(const ActionC& action) = 0;

    virtual ~LinearMAB()
    {
    }
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_MAB_LINEARMAB_H_ */
