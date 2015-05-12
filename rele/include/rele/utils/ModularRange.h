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

#ifndef INCLUDE_RELE_UTILS_MODULARRANGE_H_
#define INCLUDE_RELE_UTILS_MODULARRANGE_H_

#include "Range.h"

namespace ReLe
{


class ModularRange : public Range
{
public:
    ModularRange(const double lo, const double hi);

    virtual bool contains(const double d) const;
    virtual double bound(const double& value) const;

    virtual ~ModularRange();
};

class Range2Pi : public ModularRange
{
public:
    Range2Pi();
    static double bound(double value);


private:
    static ReLe::Range2Pi range;
};

class RangePi : public ModularRange
{
public:
    RangePi();
    static double bound(double value);
private:
    static ReLe::RangePi range;
};



}


#endif /* INCLUDE_RELE_UTILS_MODULARRANGE_H_ */
