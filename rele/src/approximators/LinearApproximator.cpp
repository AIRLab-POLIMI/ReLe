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
    basis = NULL;
}

LinearApproximator::LinearApproximator(const unsigned int input_dim, AbstractBasisVector* bfs)
    : ParametricRegressor(input_dim, 1), basis(bfs),
      parameters(bfs->size(), fill::zeros)
{
}

LinearApproximator::~LinearApproximator()
{
}

vec LinearApproximator::operator()(const vec& input)
{
    vec output(1);
    output[0] = basis->dot(input, parameters);
    return output;
}

arma::vec LinearApproximator::diff(const vec& input)
{
    AbstractBasisVector& tmpbasis = *(basis);
    return tmpbasis(input);
}

}
