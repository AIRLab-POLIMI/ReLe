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

#include "Features.h"
#include <armadillo>
#include <vector>
#include <cassert>
#include "Regressors.h"

namespace ReLe
{

template<class InputC, bool denseOutput = true>
class LinearApproximator_: public ParametricRegressor_<InputC, denseOutput>
{

public:
    LinearApproximator_(Features_<InputC, denseOutput>& bfs)
        : ParametricRegressor(bfs.cols()), basis(bfs),
          parameters(bfs.rows(), arma::fill::zeros)
    {
    }

    ~LinearApproximator_()
    {
    }

    arma::vec operator()(const InputC& input)
    {
        arma::mat features = basis(input);
        arma::vec output = features.t()*parameters;
        return output;
    }

    arma::vec diff(const InputC& input)
    {
        arma::mat features = basis(input);
        return vectorise(features);
    }

    inline Features& getBasis()
    {
        return basis;
    }

    inline arma::vec getParameters() const
    {
        return parameters;
    }

    inline void setParameters(arma::vec& params)
    {
        assert(params.n_elem == parameters.n_elem);
        parameters = params;
    }

    inline unsigned int getParametersSize() const
    {
        return parameters.n_elem;
    }

private:
    arma::vec parameters;
    Features_<InputC, denseOutput>& basis;
};

typedef LinearApproximator_<arma::vec> LinearApproximator;

} //end namespace

#endif //LINEARAPPROXIMATOR_H
