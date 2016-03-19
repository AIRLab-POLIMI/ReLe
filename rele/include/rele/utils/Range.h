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

/*!
 * Simple real-valued range. It contains an upper and lower bound.
 */
class Range
{
private:
    double lowerbound;
    double upperBound;

public:
    /*!
     * Constructor.
     * Initialize to an empty set (where lo > hi).
     */
    Range();

    /*!
    * Constructor.
    * Initialize a range to enclose only the given point (lo = point, hi = point).
    * \param point point that this range will enclose
    */
    Range(const double point);

    /*!
    * Initialize to specified range.
    * \param lo the lower bound of the range
    * \param hi the upper bound of the range
    */
    Range(const double lo, const double hi);

    /*!
     * Get the lower bound.
     * \return the lower bound value
     */
    double lo() const
    {
        return lowerbound;
    }

    /*!
     * Get a reference to the lower bound.
     * \return a reference to the lower bound
     */
    double& lo()
    {
        return lowerbound;
    }

    /*!
     * Get the upper bound.
     * \return the upper bound value
     */
    double hi() const
    {
        return upperBound;
    }

    /*!
     * Get a reference to the upper bound.
     * \return a reference to the upper bound
     */
    double& hi()
    {
        return upperBound;
    }

    /*!
    * Get the span of the range (hi - lo).
    * \return the width of the range
    */
    double width() const;

    /*!
    * Get the midpoint of this range.
    * \return the mid point value
    */
    double mid() const;

    /*!
     * Check whether a value is inside the range.
     * \param value the value to check
     * \return the lower bound in case the value is lower than it,
     * 		   the upper bound in case the value is greater than it,
     * 		   the value in case the value is inside the range
     */
    virtual double bound(const double& value) const;

    /*!
    * Expand this range to include another range.
    * \param rhs range to include
    * \return a reference to the expanded range
    */
    Range& operator|=(const Range& rhs);

    /*!
    * Expand this range to include another range.
    * \param rhs range to include
    * \return the expanded range
    */
    Range operator|(const Range& rhs) const;

    /*!
    * Shrink this range to make it overlap to another range; this makes an
    * empty set if there is no overlapping.
    * \param rhs other range
    * \return a reference to the overlapped range
    */
    Range& operator&=(const Range& rhs);

    /*!
    * Shrink this range to make it overlap to another range; this makes an
    * empty set if there is no overlapping.
    * \param rhs other range
    * \return the overlapped range
    */
    Range operator&(const Range& rhs) const;

    /*!
    * Scale the bounds by the given double.
    * \param d scaling factor
    * \return a reference to the scaled range
    */
    Range& operator*=(const double d);

    /*!
    * Scale the bounds by the given double.
    * \param d scaling factor
    * \return the scaled range
    */
    Range operator*(const double d) const;

    /*!
    * Scale the bounds of another range by a given double and creates a new range from them.
    * \param d scaling factor
    * \param r range whose bound values are to be scaled
    * \return a new scaled range
    */
    friend Range operator*(const double d, const Range& r);

    /*!
    * Compare with another range for strict equality.
    * \param rhs other range
    * \return true if the ranges are equal
    */
    bool operator==(const Range& rhs) const;

    /*!
    * Compare with another range for strict inequality.
    * param rhs other range
    * \return true if the ranges are not equal
    */
    bool operator!=(const Range& rhs) const;

    /*!
    * Compare with another range. For Range objects x and y, x < y means that x
    * is strictly less than y and does not overlap at all.
    * \param rhs other range
    * \return true if the range is strictly less than the other
    */
    bool operator<(const Range& rhs) const;

    /*!
    * Compare with another range. For Range objects x and y, x > y means that x
    * is strictly greater than y and does not overlap at all.
    * \param rhs other range
    * \return true if the range is strictly greater than the other
    */
    bool operator>(const Range& rhs) const;

    /*!
    * Determine if a point is contained within the range.
    * \param d the point to check.
    * \return true if the point is within the range
    */
    virtual bool contains(const double d) const;

    /*!
    * Determine if another range overlaps completely with this one.
    * \param r other range
    * \return true if ranges overlap completely
    */
    bool contains(const Range& r) const;

    /*!
    * Return a string representation of an object.
    * \return the string indicating lower and upper bound of the range
    */
    std::string toString() const;

    /*!
     * Print a string representation of an object to an upstream variable.
     * \param out the output where to print the string
     * \param range the range to be printed
     * \return the output object
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Range& range)
    {
        out << range.toString();
        return out;
    }

    virtual ~Range();

};

} // namespace rele


#endif //RANGE_H_
