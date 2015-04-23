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

///////////////////////////////////////
// IdentityBasis
///////////////////////////////////////

IdentityBasis::IdentityBasis(unsigned int index)
    : IdentityBasis_(index)
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

    for(unsigned int i = 0; i < input_size; i++)
    {
        basis.push_back(new IdentityBasis(i));
    }

    return basis;
}

////////////////////////////////////////////////
// FiniteIdentityBasis
////////////////////////////////////////////////

FiniteIdentityBasis::FiniteIdentityBasis(unsigned int index)
    : IdentityBasis_(index)
{

}

FiniteIdentityBasis::~FiniteIdentityBasis()
{

}

double FiniteIdentityBasis::operator() (const size_t& input)
{
	return (input == index);
}

void FiniteIdentityBasis::writeOnStream (std::ostream& out)
{
	out << "Finite Identity" << std::endl;
	out << index <<endl;
}

void FiniteIdentityBasis::readFromStream(std::istream& in)
{
	//TODO implement
}

BasisFunctions_<size_t> FiniteIdentityBasis::generate(unsigned int stateN)
{
	BasisFunctions_<size_t> basis;

	for(size_t i = 0; i < stateN; i++)
	{
		 basis.push_back(new FiniteIdentityBasis(i));
	}

	return basis;
}

}//end namespace
