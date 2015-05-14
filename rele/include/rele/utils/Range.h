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
#ifndef RANGE_H_
#define RANGE_H_

#include <iostream>

namespace ReLe
{

/**
 * Simple real-valued range.  It contains an upper and lower bound.
 */
class Range
{
private:
    double lowerbound; /// The lower bound.
    double upperBound; /// The upper bound.

public:
    /** Initialize to an empty set (where lo > hi). */
    Range();

    /***
    * Initialize a range to enclose only the given point (lo = point, hi =
    * point).
    *
    * @param point Point that this range will enclose.
    */
    Range(const double point);

    /**
    * Initializes to specified range.
    *
    * @param lo Lower bound of the range.
    * @param hi Upper bound of the range.
    */
    Range(const double lo, const double hi);

    //! Get the lower bound.
    double lo() const
    {
        return lowerbound;
    }
    //! Modify the lower bound.
    double& lo()
    {
        return lowerbound;
    }

    //! Get the upper bound.
    double hi() const
    {
        return upperBound;
    }
    //! Modify the upper bound.
    double& hi()
    {
        return upperBound;
    }

    /**
    * Gets the span of the range (hi - lo).
    */
    double width() const;

    /**
    * Gets the midpoint of this range.
    */
    double mid() const;

    virtual double bound(const double& value) const;

    /**
    * Expands this range to include another range.
    *
    * @param rhs Range to include.
    */
    Range& operator|=(const Range& rhs);

    /**
    * Expands this range to include another range.
    *
    * @param rhs Range to include.
    */
    Range operator|(const Range& rhs) const;

    /**
    * Shrinks this range to be the overlap with another range; this makes an
    * empty set if there is no overlap.
    *
    * @param rhs Other range.
    */
    Range& operator&=(const Range& rhs);

    /**
    * Shrinks this range to be the overlap with another range; this makes an
    * empty set if there is no overlap.
    *
    * @param rhs Other range.
    */
    Range operator&(const Range& rhs) const;

    /**
    * Scale the bounds by the given double.
    *
    * @param d Scaling factor.
    */
    Range& operator*=(const double d);

    /**
    * Scale the bounds by the given double.
    *
    * @param d Scaling factor.
    */
    Range operator*(const double d) const;

    /**
    * Scale the bounds by the given double.
    *
    * @param d Scaling factor.
    */
    friend Range operator*(const double d, const Range& r); // Symmetric.

    /**
    * Compare with another range for strict equality.
    *
    * @param rhs Other range.
    */
    bool operator==(const Range& rhs) const;

    /**
    * Compare with another range for strict equality.
    *
    * @param rhs Other range.
    */
    bool operator!=(const Range& rhs) const;

    /**
    * Compare with another range.  For Range objects x and y, x < y means that x
    * is strictly less than y and does not overlap at all.
    *
    * @param rhs Other range.
    */
    bool operator<(const Range& rhs) const;

    /**
    * Compare with another range.  For Range objects x and y, x < y means that x
    * is strictly less than y and does not overlap at all.
    *
    * @param rhs Other range.
    */
    bool operator>(const Range& rhs) const;

    /**
    * Determines if a point is contained within the range.
    *
    * @param d Point to check.
    */
    virtual bool contains(const double d) const;

    /**
    * Determines if another range overlaps with this one.
    *
    * @param r Other range.
    *
    * @return true if ranges overlap at all.
    */
    bool contains(const Range& r) const;

    /**
    * Returns a string representation of an object.
    */
    std::string toString() const;

    friend inline std::ostream& operator<<(std::ostream& out, const Range& range)
    {
        out << range.toString();
        return out;
    }

    virtual ~Range();

};

} // namespace rele


#endif //RANGE_H_
