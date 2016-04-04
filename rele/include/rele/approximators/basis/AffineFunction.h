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

#ifndef AFFINEFUNCTION_H_
#define AFFINEFUNCTION_H_

#include "rele/approximators/BasisFunctions.h"

namespace ReLe
{

/*!
 * This class implements affine transformation functions for an input
 * vector.
 */
class AffineFunction: public BasisFunction
{
public:
    /*!
     * Constructor.
     * \param bfs basis function
     * \param A the matrix for the affine transformation
     */
    AffineFunction(BasisFunction* bfs, arma::mat A);

    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;

    /*!
     * Return the basis functions for the affine transformation.
     * \param basis basis function
     * \param A the matrix for the affine transformation
     * \return the generated basis functions
     */
    static BasisFunctions generate(BasisFunctions& basis, arma::mat& A);

private:
    BasisFunction* basis;
    arma::mat A;
};

}//end namespace


#endif // AFFINEFUNCTION_H_
