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

#include "BasisFunctions.h"

namespace ReLe
{

class PolynomialFunction : public BasisFunction
{
public:
    PolynomialFunction(std::vector<unsigned int> dimension, std::vector<unsigned int> degree);
    PolynomialFunction(unsigned int _dimension = 0, unsigned int _degree = 0);
    virtual ~PolynomialFunction();
    double operator() (const arma::vec& input);


    virtual void writeOnStream (std::ostream& out);
    virtual void readFromStream(std::istream& in);


    /**
         * Automatically generates polynomial basis functions up to the specified degree
         * @param  degree The maximum degree of the polynomial
         * @param  input_size Number of input dimensions
         */
    //void generatePolynomialBasisFunctions(unsigned int degree, unsigned int input_size);

    /**
     * @brief Write the internal state to the stream.
     * @see WriteOnStream
     * @param out the output stream
     * @param bf an instance of basis functions
     * @return the output stream
     */
    //friend std::ostream& operator<<(std::ostream& out, DenseBasisMatrix& bf);

    /**
     * @brief Read the internal stream from a stream
     * @see ReadFromStream
     * @param in the input stream
     * @param bf an instance of basis functions
     * @return the input stream
     */
    //friend std::istream& operator>>(std::istream& in, DenseBasisMatrix& bf);

private:
    std::vector<unsigned int> dimension;
    std::vector<unsigned int> degree;
};

}//end namespace


#endif // POLYNOMIALFUNCTION_H
