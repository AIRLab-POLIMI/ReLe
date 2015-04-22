/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta & Marcello Restelli
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

#ifndef BASISFUNCTIONS_H
#define BASISFUNCTIONS_H

#include <armadillo>
#include <stdexcept>

namespace ReLe
{

template<class InputC>
class BasisFunction_
{
public:
    virtual double operator()(const InputC& input) = 0;

    /**
     * @brief Write a complete description of the instance to
     * a stream.
     * @param out the output stream
     */
    virtual void writeOnStream(std::ostream& out) = 0;

    /**
     * @brief Read the description of the basis function from
     * a file and reset the internal state according to that.
     * This function is complementary to WriteOnStream
     * @param in the input stream
     */
    virtual void readFromStream(std::istream& in) = 0;

    /**
     * @brief Write the internal state to the stream.
     * @see WriteOnStream
     * @param out the output stream
     * @param bf an instance of basis function
     * @return the output stream
     */
    friend std::ostream& operator<<(std::ostream& out, BasisFunction_<InputC>& bf)
    {
        bf.writeOnStream(out);
        return out;
    }

    /**
     * @brief Read the internal stream from a stream
     * @see ReadFromStream
     * @param in the input stream
     * @param bf an instance of basis function
     * @return the input stream
     */
    friend std::istream& operator>>(std::istream& in, BasisFunction_<InputC>& bf)
    {
        bf.readFromStream(in);
        return in;
    }

    virtual ~BasisFunction_()
    {
    }

};

typedef BasisFunction_<arma::vec> BasisFunction;

typedef std::vector<BasisFunction*> BasisFunctions;

} //end namespace

#endif //BASISFUNCTIONS_H
