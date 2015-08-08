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

#include "nn/ActivationFunctions.h"
#include "nn/Regularization.h"

namespace ReLe
{

template<class InputC, bool denseOutput = true>
class FFNeuralNetwork_: public ParametricRegressor_<InputC, denseOutput>,
    public BatchRegressor_<InputC, arma::vec>
{

public:
    enum algorithm
    {
        GradientDescend, StochasticGradientDescend, Adadelta
    };

    struct OptimizationParameters
    {
        algorithm alg;
        double alpha;
        double lambda;
        double rho;
        double epsilon;
        unsigned int maxIterations;
        unsigned int minibatchSize;
    };

public:
    FFNeuralNetwork_(Features_<InputC, denseOutput>& bfs, unsigned int neurons,
                     unsigned int outputs) :
        ParametricRegressor(outputs), basis(bfs)
    {
        layerFunction.push_back(new Sigmoid());
        layerFunction.push_back(new Linear());

        layerNeurons.push_back(neurons);
        layerNeurons.push_back(outputs);

        setupNetwork();
    }

    FFNeuralNetwork_(Features_<InputC, denseOutput>& bfs, std::vector<unsigned int>& layerNeurons,
                     std::vector<Function*>& layerFunction) :
        ParametricRegressor(layerNeurons.back()), basis(bfs), layerFunction(layerFunction), layerNeurons(layerNeurons)
    {
        setupNetwork();
    }

    ~FFNeuralNetwork_()
    {
        for (auto f : layerFunction)
        {
            delete f;
        }
    }

    arma::vec operator()(const InputC& input)
    {
        forwardComputation(input);
        return h.back();
    }

    arma::vec diff(const InputC& input)
    {
        forwardComputation(input);
        arma::vec g(layerNeurons.back(), arma::fill::ones);

        return backPropagation(g);
    }

    inline Features& getBasis()
    {
        return basis;
    }

    inline arma::vec getParameters() const
    {
        return w;
    }

    inline void setParameters(const arma::vec& params)
    {
        assert(params.n_elem == w.n_elem);
        w = params;
    }

    inline unsigned int getParametersSize() const
    {
        return w.n_elem;
    }

    void train(const BatchData<InputC, arma::vec>& dataset)
    {
        assert(dataset.size() > 0);

        switch (params.alg)
        {
        case GradientDescend:
            gradientDescend(dataset);
            break;

        case StochasticGradientDescend:
            stochasticGradientDescend(dataset);
            break;

        case Adadelta:
            adadelta(dataset);
            break;

        default:
            break;
        }

    }

    double computeJ(const BatchData<InputC, arma::vec>& dataset,
                    double lambda)
    {
        double J = 0;

        for (unsigned int i = 0; i < dataset.size(); i++)
        {
            const InputC& x = dataset.getInput(i);
            const arma::vec& y = dataset.getOutput(i);
            forwardComputation(x);
            arma::vec yhat = h.back();
            J += arma::norm(y - yhat) / 2 + lambda * Omega->cost(w);
        }

        J /= dataset.size();

        return J;
    }

    OptimizationParameters& getHyperParameters()
    {
        return params;
    }

private:
    void setupNetwork()
    {
        w = arma::vec(calculateParamSize(), arma::fill::zeros);

        h.push_back(arma::vec());

        for (unsigned int i = 0; i < layerFunction.size(); i++)
        {
            a.push_back(arma::vec(layerNeurons[i], arma::fill::zeros));
            h.push_back(arma::vec(layerNeurons[i], arma::fill::zeros));
        }

        Omega = new L2_Regularization();

        //Default parameters
        params.alg = GradientDescend;
        params.alpha = 0.1;
        params.lambda = 0.0;
        params.minibatchSize = 100;
        params.maxIterations = 10000;
    }

    unsigned int calculateParamSize()
    {
        unsigned int paramN = (basis.rows() + 1) * layerNeurons[0];

        for (unsigned int layer = 1; layer < layerNeurons.size(); layer++)
        {
            paramN += (layerNeurons[layer - 1] + 1) * layerNeurons[layer];
        }

        return paramN;
    }

    void forwardComputation(const InputC& input)
    {
        h[0] = basis(input);
        unsigned int start = 0;
        for (unsigned int layer = 0; layer < layerFunction.size(); layer++)
        {
            a[layer].zeros();

            //Add input * weight
            for (unsigned int i = 0; i < h[layer].n_elem; i++)
            {
                unsigned int end = start + layerNeurons[layer];
                const arma::vec& wi = w(arma::span(start, end - 1));
                a[layer] += wi * h[layer][i];
                start = end;
            }

            //Add bias
            unsigned int end = start + layerNeurons[layer];
            const arma::vec& wb = w(arma::span(start, end - 1));
            a[layer] += wb;
            start = end;

            //Apply layer function
            Function& f = *layerFunction[layer];
            for (unsigned int i = 0; i < a[layer].n_elem; i++)
                h[layer + 1][i] = f(a[layer][i]);
        }
    }

    arma::vec backPropagation(arma::vec g, double lambda = 0.0)
    {
        unsigned int end1 = w.n_elem - 1;
        unsigned int end2 = w.n_elem - 1;

        arma::vec gradW(w.n_elem, arma::fill::zeros);

        //Compute normalization derivative
        arma::vec&& dOmega = lambda * Omega->diff(w);

        for (unsigned int k = a.size(); k >= 1; k--)
        {
            unsigned int layer = k - 1;
            Function& f = *layerFunction[layer];

            //Convert the gradient on the layer’s output into a gradient into the pre-nonlinearity activation
            for (unsigned int i = 0; i < a[layer].n_elem; i++)
                g[i] = g[i] * f.diff(a[layer][i]);

            //Compute gradients on bias
            unsigned int start = end1 - g.size() + 1;
            gradW(arma::span(start, end1)) = g
                                             + dOmega(arma::span(start, end1));
            end1 = start - 1;

            //Compute gradients on weights
            for (unsigned int i = 0; i < g.size(); i++)
            {
                unsigned int start = end1 - h[layer].n_elem + 1;
                gradW(arma::span(start, end1)) = g[i] * h[layer]
                                                 + dOmega(arma::span(start, end1));
                end1 = start - 1;
            }

            //Propagate the gradients w.r.t. the next lower-level hidden layer’s activations
            arma::vec gn(h[layer].n_elem);
            for (unsigned int i = 0; i < gn.n_elem; i++)
            {
                unsigned int start = end2 - g.size();
                arma::vec&& Wki = w(arma::span(start, end2 - 1));
                gn[i] = as_scalar(Wki.t() * g);
                end2 = start - 1;
            }

            g = gn;
            end2 = end1;
        }

        return gradW;
    }

    void computeGradient(const BatchData<InputC, arma::vec>& dataset,
                         double lambda, arma::vec& g)
    {
        g.zeros();

        for (unsigned int i = 0; i < dataset.size(); i++)
        {
            const InputC& x = dataset.getInput(i);
            const arma::vec& y = dataset.getOutput(i);
            forwardComputation(x);
            arma::vec yhat = h.back();
            arma::vec gs = yhat - y;
            g += backPropagation(gs, lambda);
        }

        g /= static_cast<double>(dataset.size());

    }

private:

    void stochasticGradientDescend(const BatchData<InputC, arma::vec>& dataset)
    {
        arma::vec g(w.n_elem, arma::fill::zeros);
        for (unsigned k = 0; k < params.maxIterations; k++)
        {
            const BatchData<InputC, arma::vec>* miniBatch =
                dataset.getMiniBatch(params.minibatchSize);
            computeGradient(*miniBatch, params.lambda, g);
            w -= params.alpha * g;
            delete miniBatch;
        }
    }

    void gradientDescend(const BatchData<InputC, arma::vec>& dataset)
    {
        arma::vec g(w.n_elem, arma::fill::zeros);
        for (unsigned k = 0; k < params.maxIterations; k++)
        {
            computeGradient(dataset, params.lambda, g);
            w -= params.alpha * g;

            //std::cerr << "J = " << computeJ(dataset, params.lambda) << std::endl;
        }
    }

    void adadelta(const BatchData<InputC, arma::vec>& dataset)
    {
        arma::vec g(w.n_elem, arma::fill::zeros);
        arma::vec r(w.n_elem, arma::fill::zeros);
        arma::vec s(w.n_elem, arma::fill::zeros);

        for (unsigned k = 0; k < params.maxIterations; k++)
        {
            const BatchData<InputC, arma::vec>* miniBatch =
                dataset.getMiniBatch(params.minibatchSize);
            computeGradient(*miniBatch, params.lambda, g);

            r = params.rho*r + (1 - params.rho)*arma::square(g);

            arma::vec deltaW = -arma::sqrt(s+params.epsilon)/arma::sqrt(r+params.epsilon)%g;

            s = params.rho*s + (1 - params.rho)*arma::square(deltaW);

            w += deltaW;

            //std::cerr << "J = " << computeJ(dataset, params.lambda) << std::endl;

            delete miniBatch;
        }

    }

private:
    //Computation results
    std::vector<arma::vec> a;
    std::vector<arma::vec> h;

    //Network data
    std::vector<unsigned int> layerNeurons;
    std::vector<Function*> layerFunction;
    arma::vec w;
    Regularization* Omega;
    Features_<InputC, denseOutput>& basis;

    //Optimizaion data
    OptimizationParameters params;
};

typedef FFNeuralNetwork_<arma::vec> FFNeuralNetwork;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORK_H_ */
