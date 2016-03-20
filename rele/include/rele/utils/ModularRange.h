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

#include "rele/utils/Range.h"

namespace ReLe
{
/*!
 * In some cases (e.g. angle values), ranges repeats
 * over and over and one may want to convert a number into the corresponding
 * number within the given range. This class is useful to do so.
 * For instance, given a range of [-180°, 180°] and an angle of 186°,
 * the corresponding number within the range would be -174°.
 */
class ModularRange : public Range
{
public:
    /*!
     * Constructor.
     * \param lo the lower bound of the range
     * \param hi the upper bound of the range
     */
    ModularRange(const double lo, const double hi);

    /*!
     * Determine if a point has a regular value (i.e. different from infinity or NaN).
     * \param d the point to check
     * \return true if the point has a regular value
     */
    virtual bool contains(const double d) const override;

    /*!
     * Given a number, it returns the corresponding number within the range.
     * \param value the number to be converted
     * \return the converted number
     */
    virtual double bound(const double& value) const override;

    virtual ~ModularRange();
};

/*!
 * This class is used to manage fixed
 * angle ranges from 0 to 2 * pi radians (0° to 360° in degrees).
 */
class Range2Pi : public ModularRange
{
public:
    /*!
     * Constructor.
     * Create the range between 0 and 2 * pi radians.
     */
    Range2Pi();

    /*!
     * Given a number, it returns the corresponding number within the range.
     * \param value the number to be converted
     * \return the converted number
     */
    static double wrap(double value);

private:
    static const ReLe::Range2Pi range;
};

/*!
 * This class is used to manage fixed
 * angle range from 0 to pi radians (0° to 180° in degrees).
 */
class RangePi : public ModularRange
{
public:
    /*!
     * Constructor.
     * Create the range between 0 and pi radians.
     */
    RangePi();

    /*!
     * Given a number, it returns the corresponding number within the range.
     * \param value the number to be converted
     * \return the converted number
     */
    static double wrap(double value);

private:
    static const ReLe::RangePi range;
};



}


#endif /* INCLUDE_RELE_UTILS_MODULARRANGE_H_ */
