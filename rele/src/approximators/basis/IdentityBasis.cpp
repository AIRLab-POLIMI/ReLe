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

#include "basis/IdentityBasis.h"

using namespace arma;

namespace ReLe
{

IdentityBasis::IdentityBasis(unsigned int index)
    : index(index)
{

}



IdentityBasis::~IdentityBasis()
{

}

double IdentityBasis::operator()(const vec& input)
{
    return input(index);
}

void IdentityBasis::writeOnStream(std::ostream &out)
{
    out << "Identity" << std::endl;
    out << index <<endl;
}

void IdentityBasis::readFromStream(std::istream &in)
{
    //TODO Implement
}

BasisFunctions IdentityBasis::generate(unsigned int input_size)
{
    BasisFunctions basis;

    for(int i = 0; i < input_size; i++)
    {
        basis.push_back(new IdentityBasis(i));
    }

    return basis;
}

InverseBasis::InverseBasis(BasisFunction* basis) : basis(basis)
{

}

InverseBasis::~InverseBasis()
{
    delete basis;
}

double InverseBasis::operator() (const arma::vec& input)
{
    BasisFunction& bf = *basis;
    return -bf(input);
}

void InverseBasis::writeOnStream (std::ostream& out)
{
    out << "Inverse ";
    basis->writeOnStream(out);
}

void InverseBasis::readFromStream(std::istream& in)
{
    //TODO implement
}

}//end namespace
