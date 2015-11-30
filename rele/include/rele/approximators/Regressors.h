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
#include "Features.h"

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

    template<class Input1, class Input2>
    arma::vec operator()(const Input1& input1, const Input2& input2)
    {
        auto& self = *this;
        return self(vectorize(input1, input2));
    }

    template<class Input1, class Input2, class Input3>
    arma::vec operator()(const Input1& input1, const Input2& input2, const Input3& input3)
    {
        auto& self = *this;
        return self(vectorize(input1, input2, input3));
    }

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

    template<class Input1, class Input2>
    arma::vec diff(const Input1& input1, const Input2& input2)
    {
        return this->diff(vectorize(input1, input2));
    }

    template<class Input1, class Input2, class Input3>
    arma::vec diff(const Input1& input1, const Input2& input2, const Input3& input3)
    {
        return this->diff(vectorize(input1, input2, input3));
    }

    virtual ~ParametricRegressor_()
    {

    }

};
typedef ParametricRegressor_<arma::vec> ParametricRegressor;


template<class InputC, class OutputC, bool denseOutput=true>
class BatchRegressor_ : public Regressor_<InputC, denseOutput>
{

public:
    BatchRegressor_(Features_<InputC, denseOutput>& phi, unsigned int output = 1) :
        Regressor_<InputC, denseOutput>(output),
        phi(phi)
    {

    }

    virtual void train(const BatchData<InputC, OutputC>& dataset)
    {
        unsigned int N = dataset.size();

        //compute features matrix
        arma::mat features(phi.rows(), N);
        arma::mat outputs(this->outputDimension, N);
        for(int i = 0; i < N; i++)
        {
            features.col(i) = phi(dataset.getInput(i));
            outputs.col(i) = dataset.getOutput(i);
        }

        trainFeatures(features, outputs);
    }

    virtual void trainFeatures(const arma::mat& input, const arma::rowvec& output) = 0;

    virtual ~BatchRegressor_()
    {

    }

    inline Features& getBasis()
    {
        return phi;
    }

protected:
    Features_<InputC, denseOutput>& phi;
};
typedef BatchRegressor_<arma::vec, arma::vec> BatchRegressor;

template<class InputC, bool denseOutput = true>
class UnsupervisedBatchRegressor_
{

public:
    UnsupervisedBatchRegressor_(Features_<InputC, denseOutput>& phi) : phi(phi)
    {

    }

    virtual void train(const std::vector<InputC>& dataset)
    {
        unsigned int N = dataset.size();

        //compute features matrix
        arma::mat features(phi.rows(), N);
        for(int i = 0; i < N; i++)
        {
            features.col(i) = phi(dataset[i]);
        }

        trainFeatures(features);
    }

    virtual void trainFeatures(const arma::mat& features) = 0;

    virtual ~UnsupervisedBatchRegressor_()
    {

    }

protected:
    Features_<InputC, denseOutput>& phi;
};

}

#define USE_REGRESSOR_MEMBERS(InputC, OutputC, denseOutput) \
    typedef BatchRegressor_<InputC, OutputC, denseOutput> Base; \
    using Base::phi;

#endif /* REGRESSORS_H_ */
