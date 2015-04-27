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

#ifndef INCLUDE_RELE_APPROXIMATORS_TILES_H_
#define INCLUDE_RELE_APPROXIMATORS_TILES_H_

#include <armadillo>
#include <iostream>

namespace ReLe
{

template<class InputC>
class Tiles_
{
public:
    virtual unsigned int operator()(const InputC& input) = 0;
    virtual unsigned int size() = 0;

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
    friend std::ostream& operator<<(std::ostream& out, Tiles_<InputC>& bf)
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
    friend std::istream& operator>>(std::istream& in, Tiles_<InputC>& bf)
    {
        bf.readFromStream(in);
        return in;
    }

    virtual ~Tiles_()
    {
    }

};


typedef Tiles_<arma::vec> Tiles;
typedef std::vector<Tiles*> TilesVector;
template<class InputC> using TilesVector_ = std::vector<Tiles_<InputC>*>;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_TILES_H_ */
