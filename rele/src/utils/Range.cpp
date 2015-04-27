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
#include "Range.h"
#include <cfloat>
#include <sstream>

namespace ReLe
{

/**
 * Initialize the range to 0.
 */
Range::Range() :
    lowerbound(DBL_MAX), upperBound(-DBL_MAX)
{
    /* nothing else to do */
}

/**
 * Initialize a range to enclose only the given point.
 */
Range::Range(const double point) :
    lowerbound(point), upperBound(point)
{
    /* nothing else to do */
}

/**
 * Initializes the range to the specified values.
 */
Range::Range(const double lo, const double hi) :
    lowerbound(lo), upperBound(hi)
{
    /* nothing else to do */
}

/**
 * Gets the span of the range, hi - lo.  Returns 0 if the range is negative.
 */
double Range::width() const
{
    if (lowerbound < upperBound)
        return (upperBound - lowerbound);
    else
        return 0.0;
}

double Range::bound(const double& value) const
{
    return std::max(lowerbound, std::min(upperBound, value));
}

/**
 * Gets the midpoint of this range.
 */
double Range::mid() const
{
    return (upperBound + lowerbound) / 2;
}

/**
 * Expands range to include the other range.
 */
Range& Range::operator|=(const Range& rhs)
{
    if (rhs.lowerbound < lowerbound)
        lowerbound = rhs.lowerbound;
    if (rhs.upperBound > upperBound)
        upperBound = rhs.upperBound;

    return *this;
}

Range Range::operator|(const Range& rhs) const
{
    return Range((rhs.lowerbound < lowerbound) ? rhs.lowerbound : lowerbound,
                 (rhs.upperBound > upperBound) ? rhs.upperBound : upperBound);
}

/**
 * Shrinks range to be the overlap with another range, becoming an empty
 * set if there is no overlap.
 */
Range& Range::operator&=(const Range& rhs)
{
    if (rhs.lowerbound > lowerbound)
        lowerbound = rhs.lowerbound;
    if (rhs.upperBound < upperBound)
        upperBound = rhs.upperBound;

    return *this;
}

Range Range::operator&(const Range& rhs) const
{
    return Range((rhs.lowerbound > lowerbound) ? rhs.lowerbound : lowerbound,
                 (rhs.upperBound < upperBound) ? rhs.upperBound : upperBound);
}

/**
 * Scale the bounds by the given double.
 */
Range& Range::operator*=(const double d)
{
    lowerbound *= d;
    upperBound *= d;

    // Now if we've negated, we need to flip things around so the bound is valid.
    if (lowerbound > upperBound)
    {
        double tmp = upperBound;
        upperBound = lowerbound;
        lowerbound = tmp;
    }

    return *this;
}

Range Range::operator*(const double d) const
{
    double nlo = lowerbound * d;
    double nhi = upperBound * d;

    if (nlo <= nhi)
        return Range(nlo, nhi);
    else
        return Range(nhi, nlo);
}

// Symmetric case.
Range operator*(const double d, const Range& r)
{
    double nlo = r.lowerbound * d;
    double nhi = r.upperBound * d;

    if (nlo <= nhi)
        return Range(nlo, nhi);
    else
        return Range(nhi, nlo);
}

/**
 * Compare with another range for strict equality.
 */
bool Range::operator==(const Range& rhs) const
{
    return (lowerbound == rhs.lowerbound) && (upperBound == rhs.upperBound);
}

bool Range::operator!=(const Range& rhs) const
{
    return (lowerbound != rhs.lowerbound) || (upperBound != rhs.upperBound);
}

/**
 * Compare with another range.  For Range objects x and y, x < y means that x is
 * strictly less than y and does not overlap at all.
 */
bool Range::operator<(const Range& rhs) const
{
    return upperBound < rhs.lowerbound;
}

bool Range::operator>(const Range& rhs) const
{
    return lowerbound > rhs.upperBound;
}

/**
 * Determines if a point is contained within the range.
 */
bool Range::contains(const double d) const
{
    return d >= lowerbound && d <= upperBound;
}

/**
 * Determines if this range overlaps with another range.
 */
bool Range::contains(const Range& r) const
{
    return lowerbound <= r.upperBound && upperBound >= r.lowerbound;
}

/**
 * Returns a string representation of an object.
 */
std::string Range::toString() const
{
    std::ostringstream convert;
    convert << "[" << lowerbound << ", " << upperBound << "]";
    return convert.str();
}

}//end namespace ReLe
