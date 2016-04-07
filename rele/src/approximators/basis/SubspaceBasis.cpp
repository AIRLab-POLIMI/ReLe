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

#include "rele/approximators/basis/SubspaceBasis.h"
#include <cassert>

namespace ReLe
{

SubspaceBasis::SubspaceBasis(BasisFunction* basis, std::vector<arma::span>& spanVector)
    : basis(basis), spanVector(spanVector)
{
    assert(spanVector.size() >= 1);
}

SubspaceBasis::SubspaceBasis(BasisFunction* basis, const arma::span& span)
    : basis(basis)
{
    spanVector.push_back(span);
}

SubspaceBasis::~SubspaceBasis()
{
    delete basis;
}

double SubspaceBasis::operator() (const arma::vec& input)
{
    arma::vec newInput = input(spanVector[0]);

    for(int i = 1; i < spanVector.size(); i++)
    {
        auto& span = spanVector[i];
        newInput = arma::join_vert(newInput, input(span));
    }

    BasisFunction& bf = *basis;

    return bf(newInput);
}

void SubspaceBasis::writeOnStream (std::ostream& out)
{
    basis->writeOnStream(out);
    out << "on subspace:" << std::endl;

    for(auto& span : spanVector)
    {
        if(span.whole)
        {
            out << "all" << std::endl;
        }
        else
        {
            out << "[" << span.a << "," << span.b << "]" << std::endl;
        }
    }
}

void SubspaceBasis::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}

BasisFunctions SubspaceBasis::generate(BasisFunctions& basisVector, std::vector<arma::span>& spanVector)
{
    BasisFunctions newBasisVector;

    for(auto basis : basisVector)
    {
        SubspaceBasis* bf = new SubspaceBasis(basis, spanVector);
        newBasisVector.push_back(bf);
    }

    return newBasisVector;
}

BasisFunctions SubspaceBasis::generate(BasisFunctions& basisVector, arma::span span)
{
    BasisFunctions newBasisVector;

    for(auto basis : basisVector)
    {
        SubspaceBasis* bf = new SubspaceBasis(basis, span);
        newBasisVector.push_back(bf);
    }

    return newBasisVector;
}

}
