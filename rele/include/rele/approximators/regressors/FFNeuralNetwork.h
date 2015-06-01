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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORK_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORK_H_

#include "Features.h"
#include <armadillo>
#include <vector>
#include <cassert>
#include <algorithm>
#include "Regressors.h"

namespace ReLe
{

template<class InputC, bool denseOutput = true>
class FFNeuralNetwork_ :  public ParametricRegressor_<InputC, denseOutput>
{
public:
    FFNeuralNetwork_(Features_<InputC, denseOutput>& bfs, unsigned int neurons, unsigned int outputs)
        : ParametricRegressor(outputs), basis(bfs)
    {
        layerFunction.push_back(new Sigmoid());
        layerFunction.push_back(new Linear());

        layerInputs.push_back(neurons);
        layerInputs.push_back(outputs);

        w = arma::vec(calculateParamSize(), arma::fill::zeros);

    }

    ~FFNeuralNetwork_()
    {
        for(auto f : layerFunction)
        {
            delete f;
        }
    }

    arma::vec operator()(const InputC& input)
    {
        arma::vec In = basis(input);

        unsigned int start = 0;

        for(unsigned int layer = 0; layer < layerFunction.size(); layer++)
        {
            arma::vec In_1(layerInputs[layer], arma::fill::zeros);

            //Add input * weight
            for(unsigned int i = 0; i < In.n_elem; i++)
            {
                unsigned int end = start + layerInputs[layer];
                const arma::vec& wi = w(arma::span(start, end - 1));
                In_1 += wi*In[i];

                start = end;
            }

            //Add bias
            unsigned int end = start + layerInputs[layer];
            const arma::vec& wb = w(arma::span(start, end - 1));
            In_1 += wb;

            start = end;

            //Apply layer function
            Function& f = *layerFunction[layer];
            In.set_size(In_1.n_elem);

            for(unsigned int i = 0; i < In.n_elem; i++)
                In[i] = f(In_1[i]);

        }

        return In;
    }

    arma::vec diff(const InputC& input)
    {
        //FIXME implement
        return arma::vec(w.n_elem);
    }

    inline Features& getBasis()
    {
        return basis;
    }

    inline arma::vec getParameters() const
    {
        return w;
    }

    inline void setParameters(arma::vec& params)
    {
        assert(params.n_elem == w.n_elem);
        w = params;
    }

    inline unsigned int getParametersSize() const
    {
        return w.n_elem;
    }

private:
    unsigned int calculateParamSize()
    {
        unsigned int paramN = (basis.rows() + 1) * layerInputs[0];

        for(unsigned int layer = 1; layer < layerInputs.size(); layer++)
        {
            paramN += (layerInputs[layer - 1] + 1) * layerInputs[layer];
        }

        return paramN;
    }

    class Function
    {
    public:
        virtual double operator() (double x) = 0;
    };

    class Sigmoid : public Function
    {
    public:
        virtual double operator() (double x)
        {
            return 1.0 / ( 1.0 + exp(-x));
        }
    };

    class Linear : public Function
    {
    public:
        Linear(double alpha = 1) : alpha(alpha)
        {

        }

        virtual double operator() (double x)
        {
            return alpha*x;
        }

    private:
        double alpha;
    };

private:
    std::vector<unsigned int> layerInputs;
    std::vector<Function*> layerFunction;
    arma::vec w;
    Features_<InputC, denseOutput>& basis;
};

typedef FFNeuralNetwork_<arma::vec> FFNeuralNetwork;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORK_H_ */
