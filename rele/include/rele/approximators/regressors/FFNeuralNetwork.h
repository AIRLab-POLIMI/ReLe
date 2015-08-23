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
    FFNeuralNetwork_(Features_<InputC, denseOutput>& phi, unsigned int neurons,
                     unsigned int outputs) :
        ParametricRegressor(outputs), phi(phi)
    {
        layerFunction.push_back(new Sigmoid());
        layerFunction.push_back(new Linear());

        layerNeurons.push_back(neurons);
        layerNeurons.push_back(outputs);

        setupNetwork();
    }

    FFNeuralNetwork_(Features_<InputC, denseOutput>& phi,
                     std::vector<unsigned int>& layerNeurons,
                     std::vector<Function*>& layerFunction) :
        ParametricRegressor(layerNeurons.back()), phi(phi),
        layerFunction(layerFunction), layerNeurons(layerNeurons)
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
        return phi;
    }

    inline arma::vec getParameters() const
    {
        return *w;
    }

    inline void setParameters(const arma::vec& params)
    {
        assert(params.n_elem == getParametersSize());

        *w = params;
    }

    inline unsigned int getParametersSize() const
    {
        return paramSize;
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

    double computeJ(const BatchData<InputC, arma::vec>& dataset, double lambda)
    {
        double J = 0;

        for (unsigned int i = 0; i < dataset.size(); i++)
        {
            const InputC& x = dataset.getInput(i);
            const arma::vec& y = dataset.getOutput(i);
            forwardComputation(x);
            arma::vec yhat = h.back();
            J += arma::norm(y - yhat) / 2; // + lambda * Omega->cost(w);
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
        calculateParamSize();

        h.push_back(arma::vec());

        for (unsigned int i = 0; i < layerFunction.size(); i++)
        {
            a.push_back(arma::vec(layerNeurons[i], arma::fill::zeros));
            h.push_back(arma::vec(layerNeurons[i], arma::fill::zeros));
        }

        Omega = new L2_Regularization();

        //Create the network
        aux_mem = new double[paramSize];
        unsigned int input = phi.rows();

        w = new arma::vec(aux_mem, paramSize, false);

        Wvec.resize(layerNeurons.size());
        bvec.resize(layerNeurons.size());


        unsigned int start = 0;
        for (unsigned int i = 0; i < layerNeurons.size(); i++)
        {
            unsigned int output = layerNeurons[i];
            Wvec[i] = new arma::mat(aux_mem + start, output, input, false);
            start += Wvec[i]->n_elem;
            bvec[i] = new arma::vec(aux_mem + start, output, false);
            start += bvec[i]->n_elem;
            input = output;
        }

        w->ones();
        //TODO remove
        checkParameters();

        //Default parameters
        params.alg = GradientDescend;
        params.alpha = 0.1;
        params.lambda = 0.0;
        params.minibatchSize = 100;
        params.maxIterations = 10000;
    }

    void calculateParamSize()
    {
        unsigned int paramN = (phi.rows() + 1) * layerNeurons[0];

        for (unsigned int layer = 1; layer < layerNeurons.size(); layer++)
        {
            paramN += (layerNeurons[layer - 1] + 1) * layerNeurons[layer];
        }

        paramSize = paramN;
    }

    void forwardComputation(const InputC& input)
    {
        h[0] = phi(input);
        unsigned int start = 0;
        for (unsigned int layer = 0; layer < layerFunction.size(); layer++)
        {
            //Compute activation
            a[layer] = W(layer) * h[layer] + b(layer);

            //Compute neuron outputs
            Function& f = *layerFunction[layer];
            auto fun = [&](double val)
            {
                return f(val);
            };
            h[layer + 1] = a[layer];
            h[layer + 1].transform(fun);
        }
    }

    arma::vec backPropagation(arma::vec g, double lambda = 0.0)
    {
        arma::vec gradW(paramSize);
        unsigned int end = gradW.n_elem - 1;

        //Compute normalization derivative
        //arma::vec&& dOmega = lambda * Omega->diff(w);

        for (unsigned int k = a.size(); k >= 1; k--)
        {
            unsigned int layer = k - 1;

            //Convert the gradient on the layer’s output into a gradient into the pre-nonlinearity activation
            Function& f = *layerFunction[layer];
            auto fun = [&](double val)
            {
                return f.diff(val);
            };
            arma::vec df = a[layer];
            df.transform(fun);
            g = g % df;

            //Compute gradients on bias
            unsigned int start = end - b(layer).n_elem + 1;
            gradW(arma::span(start, end)) = g; // + dOmega(arma::span(start, end));
            end = start - 1;

            start = end - W(layer).n_elem + 1;
            gradW(arma::span(start, end)) = arma::vectorise(g * h[layer].t()); // + dOmega(arma::span(start, end));
            end = start - 1;

            //Propagate the gradients w.r.t. the next lower-level hidden layer’s activations
            arma::vec gn(h[layer].n_elem);

            g = W(layer).t() * g;
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
    	arma::vec& w = *this->w;
        arma::vec g(paramSize, arma::fill::zeros);
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
    	arma::vec& w = *this->w;
        arma::vec g(paramSize, arma::fill::zeros);
        for (unsigned k = 0; k < params.maxIterations; k++)
        {
            computeGradient(dataset, params.lambda, g);
            w -= params.alpha * g;

            checkParameters();//TODO levare
            //std::cerr << "J = " << computeJ(dataset, params.lambda) << std::endl;
        }
    }

    void adadelta(const BatchData<InputC, arma::vec>& dataset)
    {
    	arma::vec& w = *this->w;

        arma::vec g(paramSize, arma::fill::zeros);
        arma::vec r(paramSize, arma::fill::zeros);
        arma::vec s(paramSize, arma::fill::zeros);

        for (unsigned k = 0; k < params.maxIterations; k++)
        {
            const BatchData<InputC, arma::vec>* miniBatch =
                dataset.getMiniBatch(params.minibatchSize);
            computeGradient(*miniBatch, params.lambda, g);

            r = params.rho * r + (1 - params.rho) * arma::square(g);

            arma::vec deltaW = -arma::sqrt(s + params.epsilon)
                               / arma::sqrt(r + params.epsilon) % g;

            s = params.rho * s + (1 - params.rho) * arma::square(deltaW);

            w += deltaW;

            //std::cerr << "J = " << computeJ(dataset, params.lambda) << std::endl;

            delete miniBatch;
        }

    }

    inline arma::mat& W(unsigned int layer)
    {
        return *Wvec[layer];
    }

    inline arma::vec& b(unsigned int layer)
    {
        return *bvec[layer];
    }

    //TODO remove
    void checkParameters()
    {
    	arma::vec& params = *this->w;

        unsigned int start = 0;
        for (unsigned int layer = 0; layer < layerFunction.size(); layer++)
        {
            unsigned int end = start + W(layer).n_elem - 1;
            arma::mat Wnew = params(arma::span(start, end));
            Wnew.reshape(W(layer).n_rows, W(layer).n_cols);
            assert(arma::sum(arma::sum(W(layer) != Wnew)) == 0);

            start = end + 1;

            end = start + b(layer).n_elem - 1;
            assert(arma::sum(b(layer) != params(arma::span(start, end))) == 0);

            start = end + 1;
        }
    }

private:
    //Computation results
    std::vector<arma::vec> a;
    std::vector<arma::vec> h;

    //Network data
    std::vector<unsigned int> layerNeurons;
    std::vector<Function*> layerFunction;
    unsigned int paramSize;

    arma::vec* w;
    std::vector<arma::mat*> Wvec;
    std::vector<arma::vec*> bvec;
    double* aux_mem;

    Regularization* Omega;
    Features_<InputC, denseOutput>& phi;

    //Optimizaion data
    OptimizationParameters params;
};

typedef FFNeuralNetwork_<arma::vec> FFNeuralNetwork;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORK_H_ */
