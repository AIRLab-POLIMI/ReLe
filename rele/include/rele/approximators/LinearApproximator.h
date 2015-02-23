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
#ifndef LINEARAPPROXIMATOR_H
#define LINEARAPPROXIMATOR_H

#include "Approximators.h"
#include "BasisFunctions.h"
#include <armadillo>
#include <vector>
#include <cassert>

namespace ReLe
{

class LinearApproximator: public ParametricRegressor
{

public:
    LinearApproximator(unsigned int input_dim, unsigned int output_dim); //TODO SISTEMARE PER L'USCITA MULTI-DIM
    LinearApproximator(const unsigned int input_dim, AbstractBasisVector* bfs);
    virtual ~LinearApproximator();
    arma::vec operator()(const arma::vec& input);
    arma::vec diff(const arma::vec& input);

    inline AbstractBasisVector& getBasis()
    {
        return *basis;
    }

    inline arma::vec& getParameters()
    {
        return parameters;
    }

    inline void setParameters(arma::vec& params)
    {
        assert(params.n_elem == parameters.n_elem);
        parameters = params;
    }

private:
    arma::vec parameters;
    AbstractBasisVector* basis; // TODO estendere al caso di uscita multidimensionale (si potrebbe fare la classe BasisFunctionsMatrix
};

} //end namespace

#endif //LINEARAPPROXIMATOR_H
