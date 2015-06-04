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

#include "basis/AffineFunction.h"

using namespace arma;

namespace ReLe
{

AffineFunction::AffineFunction(BasisFunction* bfs, arma::mat A)
    :basis(bfs), A(A)
{
}

double AffineFunction::operator()(const arma::vec& input)
{
    arma::vec z = A*input;
    return (*basis)(z);
}

void AffineFunction::writeOnStream(std::ostream& out)
{
    out << "AffineFunction";
}

void AffineFunction::readFromStream(std::istream &in)
{
}

BasisFunctions AffineFunction::generate(BasisFunctions& basis, arma::mat& A)
{
    BasisFunctions newBasis;

    unsigned int nbasis = basis.size();
    for (unsigned int i = 0; i < nbasis; ++i)
    {
        newBasis.push_back(new AffineFunction(basis[i], A));
    }
}

}//end namespace
