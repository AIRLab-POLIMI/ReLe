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

#include "LinearApproximator.h"
#include <cassert>

using namespace std;
using namespace arma;

namespace ReLe
{

LinearApproximator::LinearApproximator(unsigned int input_dim, unsigned int output_dim)
    : ParametricRegressor(input_dim, output_dim)
{
    assert(output_dim == 1);
}

LinearApproximator::LinearApproximator(const unsigned int input_dim, BasisFunctions& bfs)
    : ParametricRegressor(input_dim, 1), basis(bfs),
      parameters(bfs.size(), fill::zeros)
{
}

LinearApproximator::~LinearApproximator()
{
}

void LinearApproximator::evaluate(const vec& input, vec& output)
{
    output[0] = basis.dot(input, parameters);
}

}
