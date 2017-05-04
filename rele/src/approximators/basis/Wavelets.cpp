/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#include "rele/approximators/basis/Wavelets.h"
#include <cassert>

using namespace arma;

namespace ReLe
{

Wavelets::Wavelets(WaveletType& wavelet, unsigned int k, unsigned int index)
    : wavelet(wavelet), j(0), k(k), index(index), scale(true)
{
}

Wavelets::Wavelets(WaveletType& wavelet, unsigned int j, unsigned int k, unsigned int index)
    : wavelet(wavelet), j(j), k(k), index(index), scale(false)
{
}

Wavelets::~Wavelets()
{
}

double Wavelets::operator()(const vec& input)
{
    if(scale)
    {
        double value = input(index) - k;
        return wavelet.scaling(value);
    }
    else
    {
        double value = std::pow(2,j)*input(index) - k;
        return std::pow(2,0.5*j)*wavelet.mother(value);
    }

}

BasisFunctions Wavelets::generate(WaveletType& wavelet, unsigned int index, unsigned int jMax, int maxT)
{
    BasisFunctions basis;

    for(int k = 0; k < maxT; k++)
    {
        auto* bf = new Wavelets(wavelet, k, index);
        basis.push_back(bf);
    }

    for(unsigned int j = 0; j < jMax; j++)
    {
        for(unsigned int k = 0; k < maxT*(std::pow(2,j)); k++)
        {
            auto* bf = new Wavelets(wavelet, j, k, index);
            basis.push_back(bf);
        }
    }

    return basis;
}

void Wavelets::writeOnStream(std::ostream &out)
{
    if(scale)
    {
        out << "Scaling " << k << std::endl;
    }
    else
    {
        out << "Wavelet " << j << " " << k << std::endl;
    }

}

void Wavelets::readFromStream(std::istream &in)
{
    //TODO [SERIALIZATION] implement
}


}
