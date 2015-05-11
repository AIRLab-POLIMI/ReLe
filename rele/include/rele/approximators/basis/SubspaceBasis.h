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

#ifndef INCLUDE_RELE_APPROXIMATORS_BASIS_SUBSPACEBASIS_H_
#define INCLUDE_RELE_APPROXIMATORS_BASIS_SUBSPACEBASIS_H_

#include "BasisFunctions.h"

namespace ReLe
{

class SubspaceBasis : public BasisFunction
{
public:
    SubspaceBasis(BasisFunction* basis, arma::span& span);
    SubspaceBasis(BasisFunction* basis, std::vector<arma::span>& spanVector);
    ~SubspaceBasis();

    double operator() (const arma::vec& input);
    void writeOnStream (std::ostream& out);
    void readFromStream(std::istream& in);

    static BasisFunctions generate(BasisFunctions& basisVector, std::vector<arma::span>& spanVector);
    static BasisFunctions generate(BasisFunctions& basisVector, arma::span span);

private:
    BasisFunction* basis;
    std::vector<arma::span> spanVector;
};



}

#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_SUBSPACEBASIS_H_ */
