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

#ifndef POLYNOMIALFUNCTION_H
#define POLYNOMIALFUNCTION_H

#include "rele/approximators/BasisFunctions.h"

namespace ReLe
{

class PolynomialFunction: public BasisFunction
{
public:
    PolynomialFunction();
    PolynomialFunction(std::vector<unsigned int> dimension,
                       std::vector<unsigned int> degree);
    PolynomialFunction(unsigned int _dimension, unsigned int _degree);
    virtual ~PolynomialFunction();
    double operator()(const arma::vec& input) override;

    virtual void writeOnStream(std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

    static BasisFunctions generate(unsigned int degree, unsigned int input_size);

private:
    static void generatePolynomialsPermutations(std::vector<unsigned int> deg,
            std::vector<unsigned int>& dim, BasisFunctions& basis);
    static void generatePolynomials(std::vector<unsigned int> deg,
                                    std::vector<unsigned int>& dim, unsigned int place,
                                    BasisFunctions& basis);

private:
    std::vector<unsigned int> dimension;
    std::vector<unsigned int> degree;
};

} //end namespace

#endif // POLYNOMIALFUNCTION_H
