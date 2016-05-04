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

#include "rele/approximators/BasisFunctions.h"

namespace ReLe
{

/*!
 * This class implements functions to apply
 * the given basis functions to a subset of the
 * input.
 */
class SubspaceBasis : public BasisFunction
{
public:
	/*!
	 * Constructor.
	 * \param basis basis function
	 * \param span subset selection of the input
	 */
    SubspaceBasis(BasisFunction* basis, const arma::span& span);

	/*!
	 * Constructor.
	 * \param basis basis function
	 * \param spanVector vector of subsets selection of the input
	 */
    SubspaceBasis(BasisFunction* basis, std::vector<arma::span>& spanVector);

	/*!
	 * Destructor.
	 */
    ~SubspaceBasis();

    double operator() (const arma::vec& input) override;
    void writeOnStream (std::ostream& out) override;
    void readFromStream(std::istream& in) override;

    /*!
     * Return the basis functions for the selected subsets.
     * \param basisVector vector of basis functions
     * \param spanVector vector of subsets selection of the input
     */
    static BasisFunctions generate(BasisFunctions& basisVector, std::vector<arma::span>& spanVector);

    /*!
     * Return the basis functions for the selected subset.
     * \param basisVector vector of basis functions
     * \param span subset selection of the input
     */
    static BasisFunctions generate(BasisFunctions& basisVector, arma::span span);

private:
    BasisFunction* basis;
    std::vector<arma::span> spanVector;
};



}

#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_SUBSPACEBASIS_H_ */
