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

#ifndef REGRESSORS_H_
#define REGRESSORS_H_

#include "Basics.h"
#include "BatchData.h"

namespace ReLe
{

template<class InputC, bool denseOutput = true>
class Regressor_
{

public:

    Regressor_(unsigned int output = 1) :
        outputDimension(output)
    {
    }

    virtual arma::vec operator() (const InputC& input) = 0;

    virtual ~Regressor_()
    {
    }

    int getOutputSize()
    {
        return outputDimension;
    }

protected:
    unsigned int outputDimension;
};


typedef Regressor_<arma::vec> Regressor;

template<class InputC, bool denseOutput = true>
class ParametricRegressor_: public Regressor_<InputC, denseOutput>
{
public:
    ParametricRegressor_(unsigned int output = 1) :
        Regressor_<InputC, denseOutput>(output)
    {
    }

    virtual void setParameters(const arma::vec& params) = 0;
    virtual arma::vec getParameters() const = 0;
    virtual unsigned int getParametersSize() const = 0;
    virtual arma::vec  diff(const InputC& input) = 0;

    virtual ~ParametricRegressor_()
    {

    }

};

typedef ParametricRegressor_<arma::vec> ParametricRegressor;

template<class InputC, bool denseOutput = true>
class NonParametricRegressor_: public Regressor_<InputC, denseOutput>
{
public:
    NonParametricRegressor_(unsigned int output = 1) :
        Regressor_<InputC, denseOutput>(output)
    {
    }
};

typedef NonParametricRegressor_<arma::vec> NonParametricRegressor;

template<class InputC, class OutputC>
class BatchRegressor_
{

public:
    virtual void train(const BatchData<InputC, OutputC>& dataset) = 0;

    virtual ~BatchRegressor_()
    {

    }
};

}

#endif /* REGRESSORS_H_ */
