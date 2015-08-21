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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_GAUSSIANMIXTUREMODELS_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_GAUSSIANMIXTUREMODELS_H_

#include "Regressors.h"
#include "ArmadilloPDFs.h"
#include "ArmadilloExtensions.h"

namespace ReLe
{

template<class InputC>
class GaussianRegressor_ : public ParametricRegressor_<InputC>
{
public:
    GaussianRegressor_(Features_<InputC>& phi) : ParametricRegressor_<InputC>(1), phi(phi)
    {
        unsigned int size = phi.rows();
        sigma = arma::eye(size, size);
        mu.set_size(size);
    }

    virtual void setParameters(const arma::vec& params)
    {
        mu = params;
    }

    virtual arma::vec getParameters() const
    {
        return mu;
    }

    virtual unsigned int getParametersSize() const
    {
        return mu.n_elem;
    }

    virtual arma::vec operator() (const InputC& input)
    {
        const arma::vec& value = phi(input);

        arma::vec result(1);
        result(0) = mvnpdf(value, mu, sigma);

        return result;
    }

    virtual arma::vec diff(const arma::vec& input)
    {
        arma::vec g_mean;
        arma::vec g_sigma;
        const arma::vec& value = phi(input);

        mvnpdf(value, mu, sigma, g_mean);

        return g_mean;
    }

    virtual ~GaussianRegressor_()
    {

    }

private:
    Features_<InputC>& phi;
    arma::vec mu;
    arma::mat sigma;

};

typedef GaussianRegressor_<arma::vec> GaussianRegressor;


template<class InputC>
class GaussianMixtureRegressor_ : public ParametricRegressor_<InputC>
{
public:
    GaussianMixtureRegressor_(Features_<InputC>& phi, unsigned int n, double muVariance = 1) : ParametricRegressor_<InputC>(1), phi(phi)
    {
        unsigned int size = phi.rows();

        for(unsigned int i = 0; i < n; i++)
        {
            cholSigma.push_back(arma::eye(size, size));
            mu.push_back(muVariance*arma::vec(size, arma::fill::randn));
        }

        h = arma::vec(n, arma::fill::ones)/n;

        //default precision of EM algorithm
        precision = 1e-8;

    }

    GaussianMixtureRegressor_(Features_<InputC>& phi, std::vector<arma::vec>& mu)
        : ParametricRegressor_<InputC>(1), phi(phi), mu(mu)
    {
        unsigned int size = phi.rows();
        unsigned int n = mu.size();

        for(unsigned int i = 0; i < n; i++)
        {
            cholSigma.push_back(arma::eye(size, size));
        }

        h = arma::vec(n, arma::fill::ones)/n;

        //default precision of EM algorithm
        precision = 1e-8;
    }

    GaussianMixtureRegressor_(Features_<InputC>& phi, std::vector<arma::vec>& mu, std::vector<arma::mat>& cholSigma)
        : ParametricRegressor_<InputC>(1), phi(phi), mu(mu), cholSigma(cholSigma)
    {
        unsigned int n = mu.size();
        h = arma::vec(n, arma::fill::ones)/n;

        //default precision of EM algorithm
        precision = 1e-8;
    }

    virtual void setParameters(const arma::vec& params)
    {
        unsigned int n = mu.size();
        unsigned int dim = mu[0].size();

        h = params(arma::span(0, n-1));

        unsigned int start = n;
        for(unsigned int i = 0; i < n; i++)
        {
            mu[i] = params(arma::span(start,start + dim - 1));
            start = start + dim;
            unsigned int nSigma = dim + (dim*dim-dim)/2;
            vecToTriangular(nSigma, params(arma::span(start,start + nSigma -1)), cholSigma[i]);
            start += nSigma;
        }
    }

    virtual arma::vec getParameters() const
    {
        unsigned int size = getParametersSize();
        unsigned int n = h.size();
        unsigned int dim = mu[0].size();


        arma::vec params(size);


        params(arma::span(0, n -1)) = h;

        unsigned int start = n;
        for(unsigned int i = 0; i < n; i++)
        {
            params(arma::span(start,start + dim - 1)) = mu[i];
            start = start + dim;
            unsigned int nSigma = dim + (dim*dim-dim)/2;
            arma::vec cholVec(nSigma);
            triangularToVec(nSigma, cholSigma[i], cholVec);
            params(arma::span(start,start + nSigma -1)) = cholVec;
            start += nSigma;
        }


        return params;
    }

    virtual unsigned int getParametersSize() const
    {
        unsigned int n = mu.size();
        unsigned int dim = phi.rows();
        unsigned int varPar = dim + (dim * dim - dim) / 2;
        return dim*n+varPar*n+n;
    }

    virtual arma::vec operator() (const InputC& input)
    {
        const arma::vec& value = phi(input);

        arma::vec result = {arma::sum(computeMemberships(value))};

        return result;
    }

    virtual arma::vec diff(const arma::vec& input)
    {
        arma::vec diffV(getParametersSize());

        const arma::vec& value = phi(input);
        unsigned int n = mu.size();
        unsigned int dim = phi.rows();

        unsigned int start = n;
        for(unsigned int i = 0; i < n; i++)
        {
            arma::vec g_mean;
            arma::mat dCholSigma;

            diffV(i) = mvnpdf(value, mu[i], cholSigma[i], g_mean, dCholSigma);

            diffV(arma::span(start,start + dim - 1)) = h[i]*g_mean;
            start = start + dim;
            unsigned int nSigma = dim + (dim*dim-dim)/2;
            arma::vec g_sigma(nSigma);
            triangularToVec(nSigma, dCholSigma, g_sigma);
            diffV(arma::span(start,start + nSigma -1)) = h[i]*g_sigma;
            start += nSigma;
        }


        return diffV;

        return arma::vec();
    }


    void train(const std::vector<InputC>& samples)
    {
        assert(samples.size() > 0);


        double N = samples.size();

        //compute features matrix
        arma::mat features(phi.rows(), samples.size());
        for(int i = 0; i < samples.size(); i++)
        {
            features.col(i) = phi(samples[i]);
        }


        double oldLogL, logL = logLikelihood(features);

        do
        {
            //save last log likelihood
            oldLogL = logL;

            //Expectation
            arma::mat W(features.n_cols, mu.size());
            for(unsigned int i = 0; i < features.n_cols; i++)
            {
                arma::vec&& memberships = computeMemberships(features.col(i));
                W.row(i) = memberships.t()/arma::sum(memberships);
            }

            //Maximization
            arma::vec Nk = arma::sum(W).t();
            h = Nk/N;

            std::cout << "h: " << h.t();

            for(unsigned int k = 0; k < mu.size(); k++)
            {
                auto Wk = arma::diagmat(W.col(k));
                mu[k] = arma::sum(features*Wk, 1)/Nk(k);

                arma::mat delta = features;
                delta.each_col() -= mu[k];

                arma::mat Sigma = delta*Wk*delta.t()/Nk(k);
                cholSigma[k] = arma::chol(Sigma);

                std::cout << "mu["<< k<< "]: " << mu[k].t();
            }

            //compute new log likelihood
            logL = logLikelihood(features);

            std::cout << "logL: " << logL << std::endl;
            std::cout << "---------------------------" << std::endl;

        }
        while(logL - oldLogL > precision);

    }

    virtual ~GaussianMixtureRegressor_()
    {

    }


private:
    double logLikelihood(const arma::mat& samples)
    {
        GaussianMixtureRegressor_& self = *this;
        double value = 0;

        for(int i = 0; i < samples.n_cols; i++)
        {
            auto xi = samples.col(i);
            double likelihood = arma::as_scalar(self(xi));
            value += std::log(likelihood);
        }

        return value;
    }

    arma::vec computeMemberships(const arma::vec& value)
    {
        arma::vec memberships(mu.size());
        for (unsigned int i = 0; i < mu.size(); i++)
        {
            arma::mat sigma = cholSigma[i] * cholSigma[i].t();
            memberships[i] = h[i] * mvnpdf(value, mu[i], sigma);
        }
        return memberships;
    }

private:
    Features_<InputC>& phi;
    arma::vec h;
    std::vector<arma::vec> mu;
    std::vector<arma::mat> cholSigma;

    double precision;
};

typedef GaussianMixtureRegressor_<arma::vec> GaussianMixtureRegressor;

}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_GAUSSIANMIXTUREMODELS_H_ */
