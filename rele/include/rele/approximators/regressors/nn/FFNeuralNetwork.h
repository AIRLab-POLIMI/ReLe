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

#include "rele/approximators/Features.h"
#include <armadillo>
#include <vector>
#include <cassert>
#include <algorithm>

#include "rele/approximators/Regressors.h"

#include "nn_bits/ActivationFunctions.h"
#include "nn_bits/Regularization.h"
#include "nn_bits/Optimizators.h"
#include "rele/utils/NumericalGradient.h"

#include "rele/approximators/data/BatchDataNormalization.h"

namespace ReLe
{

//TODO [OPTIMIZATION] fix NN with sparse features
template<class InputC, bool denseOutput = true>
class FFNeuralNetwork_: public ParametricRegressor_<InputC, denseOutput>,
    public BatchRegressor_<InputC, arma::vec, denseOutput>
{
    USE_PARAMETRIC_REGRESSOR_MEMBERS(InputC, arma::vec, denseOutput)

    friend Optimizator<InputC, denseOutput>;

public:

    struct OptimizationParameters
    {
        OptimizationParameters() : lambda(0), freePointers(true)
        {
            optimizator = nullptr;
            Omega = &defaultRegularization;
            normalizationF = &defaultNormalization;
            normalizationO = &defaultNormalization;
        }

        ~OptimizationParameters()
        {
            if(freePointers)
            {
                if(optimizator)
                    delete optimizator;

                if(Omega && Omega != &defaultRegularization)
                    delete Omega;

                if(normalizationF && normalizationF != &defaultNormalization)
                    delete normalizationF;

                if(normalizationO && normalizationO != &defaultNormalization)
                    delete normalizationO;
            }
        }

        //Optimization agorithm
        Optimizator<InputC, denseOutput>* optimizator;

        //Regularization class
        Regularization* Omega;

        //Normalization class
        Normalization<denseOutput>* normalizationF;
        Normalization<denseOutput>* normalizationO;

        //Regularization weight
        double lambda;

        //Set to true to avoid destruction (all pointers must be cleared by hand)
        bool freePointers;

    private:
        NoNormalization<denseOutput> defaultNormalization;
        NoRegularization defaultRegularization;
    };

public:
    FFNeuralNetwork_(Features_<InputC, denseOutput>& phi, unsigned int neurons,
                     unsigned int outputs) :
        ParametricRegressor(phi, outputs), BatchRegressor_<InputC, arma::vec, denseOutput>(phi, outputs)
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
        ParametricRegressor(layerNeurons.back()), BatchRegressor_<InputC, arma::vec, denseOutput>(phi),
        layerFunction(layerFunction), layerNeurons(layerNeurons)
    {
        setupNetwork();
    }

    virtual ~FFNeuralNetwork_()
    {
        for (auto f : layerFunction)
        {
            delete f;
        }
    }

    virtual arma::vec operator()(const InputC& input) override
    {
        const arma::vec& x = params.normalizationF->normalize(Base::phi(input));
        forwardComputation(x);
        return params.normalizationO->restore(h.back());
    }

    virtual arma::vec diff(const InputC& input) override
    {
        const arma::vec& x = params.normalizationF->normalize(Base::phi(input));
        forwardComputation(x);
        arma::vec g(layerNeurons.back(), arma::fill::ones);

        return backPropagation(params.normalizationO->rescale(g));
    }

    inline arma::vec getParameters() const override
    {
        return *w;
    }

    inline void setParameters(const arma::vec& params) override
    {
        assert(params.n_elem == getParametersSize());

        *w = params;
    }

    inline unsigned int getParametersSize() const override
    {
        return paramSize;
    }

    void trainFeatures(const BatchData& featureDataset) override
    {

        //Setup default optimizator
        if(!params.optimizator)
            params.optimizator = new GradientDescend<InputC, denseOutput>(10000, 0.1);

        //Normalize dataset
        auto&& normalizedDataset = normalizeDatasetFull(featureDataset, *params.normalizationF, *params.normalizationO, true);

        //Train model
        params.optimizator->setNet(this);
        params.optimizator->train(normalizedDataset);
    }

    virtual double computeJFeatures(const BatchData& featureDataset) override
    {
        const BatchData& normalizedDataset = normalizeDatasetFull(featureDataset, *params.normalizationF, *params.normalizationO);
        return computeJlowLevel(normalizedDataset);
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

        //Create the network
        aux_mem = new double[paramSize];
        unsigned int input = Base::phi.rows();

        // create weights and initialize randomly
        w = new arma::vec(aux_mem, paramSize, false);
        w->randn();

        Wvec.resize(layerNeurons.size());
        bvec.resize(layerNeurons.size());

        unsigned int start = 0;
        for (unsigned int i = 0; i < layerNeurons.size(); i++)
        {
            unsigned int output = layerNeurons[i];

            // Create Weight view
            Wvec[i] = new arma::mat(aux_mem + start, output, input, false);
            start += Wvec[i]->n_elem;

            // Create Bias view
            bvec[i] = new arma::vec(aux_mem + start, output, false);
            start += bvec[i]->n_elem;

            // Normalize Weight view
            arma::mat& Wmat = *Wvec[i];
            Wmat /= std::sqrt(input);

            input = output;
        }

    }

    void calculateParamSize()
    {
        unsigned int paramN = (Base::phi.rows() + 1) * layerNeurons[0];

        for (unsigned int layer = 1; layer < layerNeurons.size(); layer++)
        {
            paramN += (layerNeurons[layer - 1] + 1) * layerNeurons[layer];
        }

        paramSize = paramN;
    }

protected:
    void forwardComputation(const arma::vec& input)
    {
        h[0] = input;
        unsigned int start = 0;
        for (unsigned int layer = 0; layer < layerFunction.size(); layer++)
        {
            //Compute activation
            a[layer] = W(layer) * h[layer] + b(layer);

            //Compute neuron outputs
            Function& f = *layerFunction[layer];
            h[layer + 1] = f(a[layer]);
        }
    }

    arma::vec backPropagation(arma::vec g)
    {
        arma::vec gradW(paramSize);
        unsigned int end = gradW.n_elem - 1;

        for (unsigned int k = a.size(); k >= 1; k--)
        {
            unsigned int layer = k - 1;

            //Convert the gradient on the layers output into a gradient into the pre-nonlinearity activation
            Function& f = *layerFunction[layer];
            g = g % f.diff(a[layer]);

            //Compute gradients on bias
            unsigned int start = end - b(layer).n_elem + 1;
            gradW(arma::span(start, end)) = g;
            end = start - 1;

            start = end - W(layer).n_elem + 1;
            gradW(arma::span(start, end)) = arma::vectorise(g * h[layer].t());
            end = start - 1;

            //Propagate the gradients w.r.t. the next lower-level hidden layers activations
            arma::vec gn(h[layer].n_elem);

            g = W(layer).t() * g;
        }

        return gradW;
    }

    void computeGradient(const BatchData& featureDataset, arma::vec& g)
    {
        g.zeros();

        for(unsigned int i = 0; i < featureDataset.size(); i++)
        {
            const arma::vec& x = featureDataset.getInput(i);
            const arma::vec& y = featureDataset.getOutput(i);

            forwardComputation(x);
            arma::vec yhat = h.back();
            arma::vec gs = yhat - y;
            g += backPropagation(gs);
        }

        g /= static_cast<double>(featureDataset.size());
        g += params.lambda*params.Omega->diff(*w);
    }

    double computeJlowLevel(const BatchData& featureDataset)
    {
        double J = 0;
        unsigned int nSamples = featureDataset.size();

        for(unsigned int i = 0; i < nSamples; i++)
        {
            const arma::vec& x = featureDataset.getInput(i);
            const arma::vec& y = featureDataset.getOutput(i);

            forwardComputation(x);
            arma::vec yhat = h.back();
            J += arma::norm(y - yhat);
        }

        J /= (2 * nSamples);

        J += params.lambda*params.Omega->cost(*w);

        return J;
    }

private:
    inline arma::mat& W(unsigned int layer)
    {
        return *Wvec[layer];
    }

    inline arma::vec& b(unsigned int layer)
    {
        return *bvec[layer];
    }

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

protected:
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

    //Optimization data
    OptimizationParameters params;
};

typedef FFNeuralNetwork_<arma::vec> FFNeuralNetwork;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORK_H_ */
