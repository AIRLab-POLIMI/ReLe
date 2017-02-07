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

#include "rele/approximators/Features.h"
#include "rele/approximators/Regressors.h"

#include <armadillo>
#include <vector>
#include <cassert>

namespace ReLe
{

template<bool denseInput = true>
class LinearApproximator_: public ParametricRegressor_<denseInput>
{
public:
	DEFINE_FEATURES_TYPES(denseInput)

public:
    LinearApproximator_(unsigned int input, unsigned int output)
        : ParametricRegressor_<denseInput>(input, output),
          parameters(input*output, arma::fill::zeros)
    {
    }

    ~LinearApproximator_()
    {
    }

    arma::vec operator()(const FeaturesType& input) override
    {
    	arma::vec output(this->outputDimension);
    	for(unsigned int i = 0; i < this->outputDimension; i++)
    	{
    		unsigned int start = i*this->outputDimension;
    		unsigned int end = start + this->outputDimension -1;
    		output(i) = arma::as_scalar(input.t()*parameters.rows(start, end));
    	}

        return output;
    }

    arma::vec diff(const FeaturesType& input) override
    {
        return input;
    }

    inline arma::vec getParameters() const override
    {
        return parameters;
    }

    inline void setParameters(const arma::vec& params) override
    {
        assert(params.n_elem == parameters.n_elem);
        parameters = params;
    }

    inline unsigned int getParametersSize() const override
    {
        return parameters.n_elem;
    }

private:
    arma::vec parameters;

};

typedef LinearApproximator_<> LinearApproximator;

} //end namespace

#endif //LINEARAPPROXIMATOR_H
