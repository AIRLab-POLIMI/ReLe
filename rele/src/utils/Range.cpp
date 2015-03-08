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
    lo(DBL_MAX), hi(-DBL_MAX)
{
    /* nothing else to do */
}

/**
 * Initialize a range to enclose only the given point.
 */
Range::Range(const double point) :
    lo(point), hi(point)
{
    /* nothing else to do */
}

/**
 * Initializes the range to the specified values.
 */
Range::Range(const double lo, const double hi) :
    lo(lo), hi(hi)
{
    /* nothing else to do */
}

/**
 * Gets the span of the range, hi - lo.  Returns 0 if the range is negative.
 */
double Range::Width() const
{
    if (lo < hi)
        return (hi - lo);
    else
        return 0.0;
}

double Range::bound(const double& value) const
{
    return std::max(lo, std::min(hi, value));
}

/**
 * Gets the midpoint of this range.
 */
double Range::Mid() const
{
    return (hi + lo) / 2;
}

/**
 * Expands range to include the other range.
 */
Range& Range::operator|=(const Range& rhs)
{
    if (rhs.lo < lo)
        lo = rhs.lo;
    if (rhs.hi > hi)
        hi = rhs.hi;

    return *this;
}

Range Range::operator|(const Range& rhs) const
{
    return Range((rhs.lo < lo) ? rhs.lo : lo,
                 (rhs.hi > hi) ? rhs.hi : hi);
}

/**
 * Shrinks range to be the overlap with another range, becoming an empty
 * set if there is no overlap.
 */
Range& Range::operator&=(const Range& rhs)
{
    if (rhs.lo > lo)
        lo = rhs.lo;
    if (rhs.hi < hi)
        hi = rhs.hi;

    return *this;
}

Range Range::operator&(const Range& rhs) const
{
    return Range((rhs.lo > lo) ? rhs.lo : lo,
                 (rhs.hi < hi) ? rhs.hi : hi);
}

/**
 * Scale the bounds by the given double.
 */
Range& Range::operator*=(const double d)
{
    lo *= d;
    hi *= d;

    // Now if we've negated, we need to flip things around so the bound is valid.
    if (lo > hi)
    {
        double tmp = hi;
        hi = lo;
        lo = tmp;
    }

    return *this;
}

Range Range::operator*(const double d) const
{
    double nlo = lo * d;
    double nhi = hi * d;

    if (nlo <= nhi)
        return Range(nlo, nhi);
    else
        return Range(nhi, nlo);
}

// Symmetric case.
Range operator*(const double d, const Range& r)
{
    double nlo = r.lo * d;
    double nhi = r.hi * d;

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
    return (lo == rhs.lo) && (hi == rhs.hi);
}

bool Range::operator!=(const Range& rhs) const
{
    return (lo != rhs.lo) || (hi != rhs.hi);
}

/**
 * Compare with another range.  For Range objects x and y, x < y means that x is
 * strictly less than y and does not overlap at all.
 */
bool Range::operator<(const Range& rhs) const
{
    return hi < rhs.lo;
}

bool Range::operator>(const Range& rhs) const
{
    return lo > rhs.hi;
}

/**
 * Determines if a point is contained within the range.
 */
bool Range::Contains(const double d) const
{
    return d >= lo && d <= hi;
}

/**
 * Determines if this range overlaps with another range.
 */
bool Range::Contains(const Range& r) const
{
    return lo <= r.hi && hi >= r.lo;
}

/**
 * Returns a string representation of an object.
 */
std::string Range::ToString() const
{
    std::ostringstream convert;
    convert << "[" << lo << ", " << hi << "]";
    return convert.str();
}

}//end namespace ReLe
