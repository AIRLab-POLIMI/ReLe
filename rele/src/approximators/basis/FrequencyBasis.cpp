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

#include "rele/approximators/basis/FrequencyBasis.h"
#include <cassert>

using namespace arma;

namespace ReLe
{
FrequencyBasis::FrequencyBasis(double f, double phi, unsigned int index)
    : omega(2.0*f*M_PI), phi(phi), index(index)
{
}

FrequencyBasis::~FrequencyBasis()
{
}

double FrequencyBasis::operator()(const vec& input)
{
	return std::sin(omega*input(index)+phi);
}

BasisFunctions FrequencyBasis::generate(unsigned int index, double fS, double fE, double df, double phi)
{
	BasisFunctions basis;

	for(double f = fS; f <= fE; f += df)
	{
		auto* bf = new FrequencyBasis(f, phi, index);
		basis.push_back(bf);
	}

	return basis;
}

BasisFunctions FrequencyBasis::generate(unsigned int index, double fS, double fE, double df, bool sine)
{
	return generate(index, fS, fE, df, sine ? 0.0 : 0.5*M_PI);
}

void FrequencyBasis::writeOnStream(std::ostream &out)
{
    out << "FrequancyBasis " << omega << " " << phi << std::endl;
}

void FrequencyBasis::readFromStream(std::istream &in)
{
	//TODO [SERIALIZATION] implement
}


}
