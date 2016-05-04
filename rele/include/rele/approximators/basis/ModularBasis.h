/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_APPROXIMATORS_BASIS_MODULARBASIS_H_
#define INCLUDE_RELE_APPROXIMATORS_BASIS_MODULARBASIS_H_

#include "rele/approximators/BasisFunctions.h"
#include "rele/utils/ModularRange.h"

#include <functional>

namespace ReLe
{

/*!
 * This class is the interface to build basis functions
 * that transform the input into the corresponding value
 * in the given range.
 */
class ModularBasis : public BasisFunction
{
public:
    /*!
     * Constructor.
     * \param index1 index of the first element of the input
     * \param index2 index of the second element of the input
     * \param range range to be used to transform the input
     */
    ModularBasis(unsigned int index1, unsigned int index2, const ModularRange& range);

protected:
    unsigned int index1;
    unsigned int index2;
    const ModularRange& range;
};

/*!
 * This class builds basis functions to transform the
 * sum of the elements of the input at the given indexes
 * into the corresponding value in the given range.
 */
class ModularSum : public ModularBasis
{
public:
    /*!
     * Constructor.
     * \param index1 index of the first element of the input
     * \param index2 index of the second element of the input
     * \param range range to be used to transform the input
     */
    ModularSum(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

/*!
 * This class builds basis functions to transform the
 * difference of the elements of the input at the given indexes
 * into the corresponding value in the given range.
 */
class ModularDifference : public ModularBasis
{
public:
    /*!
     * Constructor.
     * \param index1 index of the first element of the input
     * \param index2 index of the second element of the input
     * \param range range to be used to transform the input
     */
    ModularDifference(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

/*!
 * This class builds basis functions to transform the
 * product of the elements of the input at the given indexes
 * into the corresponding value in the given range.
 */
class ModularProduct : public ModularBasis
{
public:
    /*!
     * Constructor.
     * \param index1 index of the first element of the input
     * \param index2 index of the second element of the input
     * \param range range to be used to transform the input
     */
    ModularProduct(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

/*!
 * This class builds basis functions to transform the
 * division of the elements of the input at the given indexes
 * into the corresponding value in the given range.
 */
class ModularDivision : public ModularBasis
{
public:
    /*!
     * Constructor.
     * \param index1 index of the first element of the input
     * \param index2 index of the second element of the input
     * \param range range to be used to transform the input
     */
    ModularDivision(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_MODULARBASIS_H_ */
