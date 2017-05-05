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

#include "rele/approximators/basis/MeyerWavelets.h"
#include <cassert>

using namespace arma;

namespace ReLe
{


MeyerWavelets::~MeyerWavelets()
{
}

double MeyerWavelets::scaling(double value)
{
	if(std::abs(value) == 0.75)
	{
		return 2.0/3.0/M_PI;
	}
	else if(value != 0)
    {
    	double tmp1 = std::sin(2*M_PI/3.0*value);
    	double tmp2 = 4.0/3.0*value*std::cos(4*M_PI/3.0*value);
    	double tmp3 = M_PI*value - 16.0/9.0*M_PI*value*value*value;

    	return (tmp1+tmp2)/tmp3;
    }

    return 2.0/3.0+4.0/3.0/M_PI;
}

double MeyerWavelets::mother(double value)
{
	double tmp1 = 4.0/3.0/M_PI*(value-0.5)*std::cos(2*M_PI/3.0*(value-0.5));
	double tmp2 = std::sin(4.0*M_PI/3.0*(value-0.5))/M_PI;
	double tmp3 = value - 0.5 - 16.0/9.0*std::pow(value -0.5, 3);
	double phi1 = (tmp1-tmp2)/tmp3;


	tmp1 = 8.0/3.0/M_PI*(value-0.5)*std::cos(8*M_PI/3.0*(value-0.5));
	tmp2 = std::sin(4.0*M_PI/3.0*(value-0.5))/M_PI;
	tmp3 = value - 0.5 - 64.0/9.0*std::pow(value -0.5, 3);
	double phi2 = (tmp1+tmp2)/tmp3;

    return phi1+phi2;
}


}
