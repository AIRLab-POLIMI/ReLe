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

#include "rele/approximators/basis/NormBasis.h"

using namespace std;
using namespace arma;

namespace ReLe
{


NormBasis::NormBasis(unsigned int p) : p(p)
{

}

double NormBasis::operator() (const vec& input)
{
    return norm(input, p);
}

void NormBasis::writeOnStream (ostream& out)
{
    cout << "Norm L" << p << " basis" << std::endl;
}

void NormBasis::readFromStream(istream& in)
{
    //TODO [SERIALIZATION] implement
}

InfiniteNorm::InfiniteNorm(bool max)
{
    if(max)
        type = "inf";
    else
        type = "-inf";
}

double InfiniteNorm::operator() (const arma::vec& input)
{
    return arma::norm(input, type.c_str());
}

void InfiniteNorm::writeOnStream (std::ostream& out)
{
    if(type == "inf")
        cout << "Infinite norm basis" << endl;
    else
        cout << "Minus infinite norm basis" << endl;
}

void InfiniteNorm::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}



}
