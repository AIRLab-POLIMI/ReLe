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

namespace ReLe
{
/*!
 * This interface represents a basis function.
 * A basis function can be see as a mapping \f$\psi(i)\rightarrow\mathbb{R}\f$ with \f$i\in\mathcal{D}\f$.
 * The template definition allows for generic domains \f$\mathcal{D}\f$.
 * A set of basis function can be used as a set of features, that can be used for function approximation.
 */
template<class InputC>
class BasisFunction_
{
public:
    /*!
     * Evaluates the basis function in input.
     * \param input the input data
     * \return the value of the basis function at the input
     */
    virtual double operator()(const InputC& input) = 0;

    /*!
     * Writes the basis function to stream.
     */
    virtual void writeOnStream(std::ostream& out) = 0;

    /*!
     * Reads the basis function from stream.
     */
    virtual void readFromStream(std::istream& in) = 0;

    /*!
     * Writes the basis function to stream.
     */
    friend std::ostream& operator<<(std::ostream& out, BasisFunction_<InputC>& bf)
    {
        bf.writeOnStream(out);
        return out;
    }

    /*!
     * Reads the basis function from stream.
     */
    friend std::istream& operator>>(std::istream& in, BasisFunction_<InputC>& bf)
    {
        bf.readFromStream(in);
        return in;
    }

    /*!
     * Destructor.
     */
    virtual ~BasisFunction_()
    {
    }

};

//! Template alias.
typedef BasisFunction_<arma::vec> BasisFunction;

//! Template alias.
typedef std::vector<BasisFunction*> BasisFunctions;

//! Alias for generic basis function vector.
template<class InputC> using BasisFunctions_ = std::vector<BasisFunction_<InputC>*>;

} //end namespace

#endif //BASISFUNCTIONS_H
