/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_NN_BITS_OPTIMIZATORS_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_NN_BITS_OPTIMIZATORS_H_

#define USE_OPTIMIZATOR_METHODS(InputC, denseOutput) \
	typedef Optimizator<InputC, denseOutput> Base; \
    using Base::maxIterations; \
	using Base::w; \
	using Base::computeGradient; \
	using Base::computeJlowLevel;

namespace ReLe
{

template<class InputC, bool denseOutput>
class FFNeuralNetwork_;

template<class InputC, bool denseOutput = true>
class Optimizator
{
    friend FFNeuralNetwork_<InputC, denseOutput>;

public:
    Optimizator(unsigned int maxIterations) : maxIterations(maxIterations)
    {
        net = nullptr;
        w = nullptr;
    }

    virtual void train(const BatchData& featureDataset) = 0;

    virtual ~Optimizator()
    {

    }

protected:
    inline void computeGradient(const BatchData& featureDataset, arma::vec& g)
    {
        net->computeGradient(featureDataset, g);
    }

    inline double computeJlowLevel(const BatchData& featureDataset)
    {
        return net->computeJlowLevel(featureDataset);
    }

private:
    void setNet(FFNeuralNetwork_<InputC, denseOutput>* net)
    {
        this->net = net;
        this->w = net->w;
    }

private:
    FFNeuralNetwork_<InputC, denseOutput>* net;

protected:
    arma::vec* w;
    unsigned int maxIterations;

};

template<class InputC, bool denseOutput = true>
class GradientDescend: public Optimizator<InputC, denseOutput>
{
    USE_OPTIMIZATOR_METHODS(InputC, denseOutput)
public:
    GradientDescend(unsigned int maxIterations, double alpha)
        : Optimizator<InputC, denseOutput>(maxIterations), alpha(alpha)
    {

    }

    virtual void train(const BatchData& featureDataset) override
    {
        arma::vec& w = *this->w;
        arma::vec g(w.n_elem, arma::fill::zeros);

        for (unsigned k = 0; k < maxIterations; k++)
        {
            computeGradient(featureDataset, g);
            w -= alpha * g;
        }
    }

    virtual ~GradientDescend()
    {

    }

private:
    double alpha;
};

template<class InputC, bool denseOutput = true>
class StochasticGradientDescend: public Optimizator<InputC, denseOutput>
{
    USE_OPTIMIZATOR_METHODS(InputC, denseOutput)
public:
    StochasticGradientDescend(unsigned int maxIterations, double alpha, unsigned int minibatchSize)
        : Optimizator<InputC, denseOutput>(maxIterations), alpha(alpha), minibatchSize(minibatchSize)
    {

    }

    virtual void train(const BatchData& featureDataset) override
    {
        arma::vec& w = *this->w;
        arma::vec g(w.n_elem, arma::fill::zeros);
        for (unsigned k = 0; k < maxIterations; k++)
        {
            for(auto* miniBatch : featureDataset.getMiniBatches(minibatchSize))
            {
                computeGradient(*miniBatch, g);
                w -= alpha * g;
                delete miniBatch;
            }
        }
    }

    virtual ~StochasticGradientDescend()
    {

    }

private:
    double alpha;
    unsigned int minibatchSize;

};

template<class InputC, bool denseOutput = true>
class Adadelta: public Optimizator<InputC, denseOutput>
{
    USE_OPTIMIZATOR_METHODS(InputC, denseOutput)

public:
    Adadelta(unsigned int maxIterations, unsigned int minibatchSize, double rho, double epsilon)
        : Optimizator<InputC, denseOutput>(maxIterations), minibatchSize(minibatchSize), rho(rho), epsilon(epsilon)
    {

    }

    virtual void train(const BatchData& featureDataset) override
    {
        arma::vec& w = *this->w;

        arma::vec g(w.n_elem, arma::fill::zeros);
        arma::vec r(w.n_elem, arma::fill::zeros);
        arma::vec s(w.n_elem, arma::fill::zeros);

        for (unsigned k = 0; k < maxIterations; k++)
        {

            for(auto* miniBatch : featureDataset.getMiniBatches(minibatchSize))
            {
                computeGradient(*miniBatch, g);

                r = rho * r + (1 - rho) * arma::square(g);

                arma::vec deltaW = -arma::sqrt(s + epsilon)
                                   / arma::sqrt(r + epsilon) % g;

                s = rho * s + (1 - rho) * arma::square(deltaW);

                w += deltaW;

                delete miniBatch;

            }

        }
    }


    virtual ~Adadelta()
    {

    }

private:
    unsigned int minibatchSize;
    double rho;
    double epsilon;

};

template<class InputC, bool denseOutput = true>
class ScaledConjugateGradient: public Optimizator<InputC, denseOutput>
{
    USE_OPTIMIZATOR_METHODS(InputC, denseOutput)
public:
    ScaledConjugateGradient(unsigned int maxIterations)
        : Optimizator<InputC, denseOutput>(maxIterations)
    {

    }

    virtual void train(const BatchData& featureDataset) override
    {
        //init weights
        arma::vec& w = *this->w;
        arma::vec wOld;

        //init parameters;
        double l = 5e-7;
        double lBar = 0;
        double sigmaPar = 5e-5;

        //compute initial error
        double errorOld = computeJlowLevel(featureDataset);

        //Compute first gradient
        arma::vec g(w.n_elem, arma::fill::zeros);
        computeGradient(featureDataset, g);

        //first order info
        arma::vec r = -g;
        arma::vec p = r;
        double pNorm2 = arma::as_scalar(p.t()*p);

        //second order info
        arma::vec s;
        double delta;

        bool success = true;

        for (unsigned k = 1; k < maxIterations +1; k++)
        {
            // save current parameters
            wOld = w;

            // calculate second order information
            if(success)
            {
                double sigma = sigmaPar/std::sqrt(pNorm2);

                w += sigma*p;

                arma::vec gn(w.n_elem, arma::fill::zeros);
                computeGradient(featureDataset, gn);

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
            double error = computeJlowLevel(featureDataset);
            double Delta = 2*delta*(errorOld - error)/std::pow(mu, 2);

            // if Delta >= 0 a reduction in error can be made
            if(Delta >= 0)
            {
                computeGradient(featureDataset, g);
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


    virtual ~ScaledConjugateGradient()
    {

    }
};



}



#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_NN_BITS_OPTIMIZATORS_H_ */
