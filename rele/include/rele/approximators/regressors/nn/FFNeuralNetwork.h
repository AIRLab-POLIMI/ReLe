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

#include "nn_bits/ActivationFunctions.h"
#include "nn_bits/Regularization.h"
#include "nn_bits/Normalization.h"

#include "NumericalGradient.h"

namespace ReLe
{

//FIXME fix NN with sparse features
template<class InputC, bool denseOutput = true>
class FFNeuralNetwork_: public ParametricRegressor_<InputC, denseOutput>,
    public BatchRegressor_<InputC, arma::vec, denseOutput>
{
    USE_PARAMETRIC_REGRESSOR_MEMBERS(InputC, arma::vec, denseOutput)

public	:
    enum algorithm
    {
        GradientDescend, StochasticGradientDescend, Adadelta, ScaledConjugateGradient
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

        delete Omega;
        delete normalization;

    }

    virtual arma::vec operator()(const InputC& input) override
    {
        const arma::vec& x = normalization->normalizeInput(Base::phi(input));
        forwardComputation(x);
        return h.back();
    }

    virtual arma::vec diff(const InputC& input) override
    {
        const arma::vec& x = normalization->normalizeInput(Base::phi(input));
        forwardComputation(x);
        arma::vec g(layerNeurons.back(), arma::fill::ones);

        return backPropagation(g);
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

    void trainFeatures(BatchDataFeatures& featureDataset) override
    {
        //Clean old normalization
        /*if(normalization)
         delete normalization;

         //Compute dataset normalization
         normalization = new MinMaxNormalization(featureDataset);*/

        switch (params.alg)
        {
        case GradientDescend:
            gradientDescend(featureDataset);
            break;

        case StochasticGradientDescend:
            stochasticGradientDescend(featureDataset);
            break;

        case Adadelta:
            adadelta(featureDataset);
            break;

        case ScaledConjugateGradient:
            scaledConjugateGradient(featureDataset);
            break;

        default:
            break;
        }

    }

    double computeJFeatures(const BatchDataFeatures& featureDataset, double lambda)
    {
        double J = 0;
        unsigned int nSamples = featureDataset.size();

        for(unsigned int i = 0; i < nSamples; i++)
        {
            const arma::vec& x = normalization->normalizeInput(featureDataset.getInput(i));
            const arma::vec& y = featureDataset.getOutput(i);

            forwardComputation(x);
            arma::vec yhat = h.back();
            J += arma::norm(y - yhat);
        }

        J /= (2 * nSamples);

        return J;
    }

    //TODO rewrite using computeJfeatures
    double computeJ(const BatchDataRaw_<InputC, arma::vec>& dataset, double lambda)
    {
        double J = 0;

        for (unsigned int i = 0; i < dataset.size(); i++)
        {
            const InputC& x = normalization->normalizeInput(Base::phi(dataset.getInput(i)));
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
        normalization = new NoNormalization();

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

        //Default parameters
        params.alg = GradientDescend;
        params.alpha = 0.1;
        params.lambda = 0.0;
        params.maxIterations = 10000;
        params.minibatchSize = 100;
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
            gradW(arma::span(start, end)) = g;// + dOmega(arma::span(start, end));
            end = start - 1;

            start = end - W(layer).n_elem + 1;
            gradW(arma::span(start, end)) = arma::vectorise(g * h[layer].t());// + dOmega(arma::span(start, end));
            end = start - 1;

            //Propagate the gradients w.r.t. the next lower-level hidden layers activations
            arma::vec gn(h[layer].n_elem);

            g = W(layer).t() * g;
        }

        return gradW;
    }

    void computeGradient(const BatchData& featureDataset,
                         double lambda, arma::vec& g)
    {
        g.zeros();

        for(unsigned int i = 0; i < featureDataset.size(); i++)
        {
            const arma::vec& x = normalization->normalizeInput(featureDataset.getInput(i));
            const arma::vec& y = featureDataset.getOutput(i);

            forwardComputation(x);
            arma::vec yhat = h.back();
            arma::vec gs = yhat - y;
            g += backPropagation(gs);
        }

        g /= static_cast<double>(featureDataset.size());
        g += lambda*Omega->diff(*w);
    }

    void computeGradientNumerical(const BatchDataRaw_<InputC, arma::vec>& dataset, arma::vec& g)
    {
        g.zeros();

        for (unsigned int i = 0; i < dataset.size(); i++)
        {
            const InputC& x = dataset.getInput(i);
            const arma::vec& y = dataset.getOutput(i);

            FFNeuralNetwork_& net = *this;

            arma::vec wold = net.getParameters();

            auto lambda = [&](const arma::vec& par)
            {

                net.setParameters(par);
                double value = arma::as_scalar(net(x) - y);

                return 0.5*value*value;

            };

            g += NumericalGradient::compute(lambda, wold);

            net.setParameters(wold);
        }

        g /= static_cast<double>(dataset.size());

    }

private:

    void gradientDescend(const BatchDataFeatures& featureDataset)
    {
        arma::vec& w = *this->w;
        arma::vec g(paramSize, arma::fill::zeros);

        for (unsigned k = 0; k < params.maxIterations; k++)
        {
            computeGradient(featureDataset, params.lambda, g);
            w -= params.alpha * g;

            //std::cout << computeJFeatures(featureDataset, 0) << std::endl;
        }
    }

    void stochasticGradientDescend(const BatchDataFeatures& featureDataset)
    {
        arma::vec& w = *this->w;
        arma::vec g(paramSize, arma::fill::zeros);
        for (unsigned k = 0; k < params.maxIterations; k++)
        {
            for(auto* miniBatch : featureDataset.getMiniBatches(params.minibatchSize))
            {
                computeGradient(*miniBatch, params.lambda, g);
                w -= params.alpha * g;
                delete miniBatch;
            }

            //std::cout << computeJFeatures(featureDataset, 0) << std::endl;
        }
    }

    void adadelta(const BatchDataFeatures& featureDataset)
    {
        arma::vec& w = *this->w;

        arma::vec g(paramSize, arma::fill::zeros);
        arma::vec r(paramSize, arma::fill::zeros);
        arma::vec s(paramSize, arma::fill::zeros);

        for (unsigned k = 0; k < params.maxIterations; k++)
        {

            for(auto* miniBatch : featureDataset.getMiniBatches(params.minibatchSize))
            {
                computeGradient(*miniBatch, params.lambda, g);

                r = params.rho * r + (1 - params.rho) * arma::square(g);

                arma::vec deltaW = -arma::sqrt(s + params.epsilon)
                                   / arma::sqrt(r + params.epsilon) % g;

                s = params.rho * s + (1 - params.rho) * arma::square(deltaW);

                w += deltaW;

                delete miniBatch;

            }

        }

    }

    void scaledConjugateGradient(const BatchDataFeatures& featureDataset)
    {
        //init weights
        arma::vec& w = *this->w;
        arma::vec wOld;

        //init parameters;
        double l = 5e-7;
        double lBar = 0;
        double sigmaPar = 5e-5;

        //compute initial error
        double errorOld = computeJFeatures(featureDataset, params.lambda);

        //Compute first gradient
        arma::vec g(paramSize, arma::fill::zeros);
        computeGradient(featureDataset, params.lambda, g);

        //first order info
        arma::vec r = -g;
        arma::vec p = r;
        double pNorm2 = arma::as_scalar(p.t()*p);

        //second order info
        arma::vec s;
        double delta;

        bool success = true;

        for (unsigned k = 1; k < params.maxIterations +1; k++)
        {
            // save current parameters
            wOld = w;

            // calculate second order information
            if(success)
            {
                double sigma = sigmaPar/std::sqrt(pNorm2);

                w += sigma*p;

                arma::vec gn(paramSize, arma::fill::zeros);
                computeGradient(featureDataset, params.lambda, gn);

                s = (gn - g)/sigma;
                delta = arma::as_scalar(p.t()*s);
            }

            // scale delta
            delta += (l - lBar)*pNorm2;

            // if delta <= 0 make the hessian positive definite
            if(delta <= 0)
            {
                lBar = 2*(l - delta/pNorm2);
                delta = -delta+l*pNorm2;
                l = lBar;
            }

            // calculate step size
            double mu = arma::as_scalar(p.t()*r);
            double alfa = mu/delta;

            // calculate comparison parameter
            w = wOld + alfa*p;
            double error = computeJFeatures(featureDataset, params.lambda);
            double Delta = 2*delta*(errorOld - error)/std::pow(mu, 2);

            // if Delta >= 0 a reduction in error can be made
            if(Delta >= 0)
            {
                computeGradient(featureDataset, params.lambda, g);
                arma::vec rn = -g;

                lBar = 0;
                success = true;

                // restart algorithm if needed else update p
                if(k % w.n_elem == 0)
                {
                    p = rn;
                    pNorm2 = arma::as_scalar(p.t()*p);
                }
                else
                {
                    double beta = arma::as_scalar(rn.t()*rn - rn.t()*r);
                    p = rn + beta*p;
                    pNorm2 = arma::as_scalar(p.t()*p);
                }

                r = rn;

                // if Delta >= 0.75 reduce scale parameter
                if(Delta >= 0.75)
                    l = l/4;
            }
            else
            {
                w = wOld; //restore previous parameters
                lBar = l;
                success = false;
            }

            if(Delta < 0.25)
                l += (delta*(1 - Delta)/pNorm2);

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

    //Regularization class
    Regularization* Omega;
    Normalization* normalization;

    //Optimization data
    OptimizationParameters params;
};

typedef FFNeuralNetwork_<arma::vec> FFNeuralNetwork;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORK_H_ */
