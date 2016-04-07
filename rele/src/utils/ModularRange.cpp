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

#include "rele/utils/ModularRange.h"
#include <cmath>

namespace ReLe
{


ModularRange::ModularRange(const double lo, const double hi) : Range(lo, hi)
{

}

bool ModularRange::contains(const double d) const
{
    return !std::isnan(d) && !std::isinf(d);
}

double ModularRange::bound(const double& value) const
{
    if(contains(value))
    {
        double tmp = value - lo();
        tmp = std::fmod(tmp, width());
        return tmp + lo();
    }
    else
    {
        return value;
    }
}

ModularRange::~ModularRange()
{

}

Range2Pi::Range2Pi() : ModularRange(0.0, 2*M_PI)
{

}

const Range2Pi Range2Pi::range;

double Range2Pi::wrap(double value)
{
    return range.bound(value);
}

RangePi::RangePi() : ModularRange(-M_PI, M_PI)
{

}

double RangePi::wrap(double value)
{
    return range.bound(value);
}

const RangePi RangePi::range;

}
