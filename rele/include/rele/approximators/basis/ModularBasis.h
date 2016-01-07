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


class ModularBasis : public BasisFunction
{
public:
    ModularBasis(unsigned int index1, unsigned int index2, const ModularRange& range);

protected:
    unsigned int index1;
    unsigned int index2;
    const ModularRange& range;
};

class ModularSum : public ModularBasis
{
public:
    ModularSum(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

class ModularDifference : public ModularBasis
{
public:
    ModularDifference(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

class ModularProduct : public ModularBasis
{
public:
    ModularProduct(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

class ModularDivision : public ModularBasis
{
public:
    ModularDivision(unsigned int index1, unsigned int index2, const ModularRange& range);
    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_MODULARBASIS_H_ */
