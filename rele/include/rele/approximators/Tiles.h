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
#include <stdexcept>

namespace ReLe
{

class out_of_bounds : public std::exception
{

};

/*!
 * This class is a common interface for tiles.
 * Tiles are a way to divide an input space in multiple portions.
 * In other words, tiles implements a discretization over the input space.
 * Formally this class implements a mapping from an input data to a
 * positive number, which represent the corresponding tile.
 * One or a set of tilings can be used as a set of features for approximators.
 */
template<class InputC>
class Tiles_
{
public:
    /*!
     * Find the index of the corresponding input tile.
     */
    virtual unsigned int operator()(const InputC& input) = 0;

    /*!
     * Getter.
     * \return the total number of tiles
     */
    virtual unsigned int size() = 0;

    /*!
     * Writes the basis function to stream
     */
    virtual void writeOnStream(std::ostream& out) = 0;

    /*!
     * Read the basis function from stream
     */
    virtual void readFromStream(std::istream& in) = 0;

    /*!
     * Writes the basis function to stream
     */
    friend std::ostream& operator<<(std::ostream& out, Tiles_<InputC>& bf)
    {
        bf.writeOnStream(out);
        return out;
    }

    /*!
     * Read the basis function from stream
     */
    friend std::istream& operator>>(std::istream& in, Tiles_<InputC>& bf)
    {
        bf.readFromStream(in);
        return in;
    }

    /*!
     * Destructor.
     */
    virtual ~Tiles_()
    {
    }

};

//! Template alias.
typedef Tiles_<arma::vec> Tiles;

//! Template alias.
typedef std::vector<Tiles*> TilesVector;

//! Alias for generic tiles vector.
template<class InputC> using TilesVector_ = std::vector<Tiles_<InputC>*>;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_TILES_H_ */
