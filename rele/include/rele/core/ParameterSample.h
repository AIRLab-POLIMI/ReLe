/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef PARAMETERSAMPLE_H_
#define PARAMETERSAMPLE_H_

#include <armadillo>

namespace ReLe
{

struct ParameterSample
{
    ParameterSample(const arma::vec& theta, double r) :
        theta(theta), r(r)
    {

    }

    arma::vec theta;
    double r;
};

}

#endif /* PARAMETERSAMPLE_H_ */
