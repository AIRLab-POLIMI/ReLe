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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_OTHERS_GAUSSIANPROCESS_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_OTHERS_GAUSSIANPROCESS_H_

#include "rele/approximators/Features.h"
#include "rele/approximators/Regressors.h"
#include "rele/utils/ArmadilloExtensions.h"
#include <boost/math/distributions/normal.hpp>

namespace ReLe
{

/*!
 * Gaussian Processes are functions in continuous space domains where each point
 * is associated with a normally distributed random variable. Every finite set
 * of these random variables has a multivariate normal distribution.
 * Gaussian Processes can be used for regression starting from a prior distribution
 * and computing a posterior distribution w.r.t. training examples in a dataset.
 * The approximated function returns the mean value of the Gaussian Process w.r.t.
 * the given input.
 */

template<class InputC, bool denseOutput = true>
class GaussianProcess_ : public BatchRegressor_<InputC, arma::vec, denseOutput>
{
public:
    struct HyperParameters
    {
        HyperParameters() : lengthScale({1}),
                        signalSigma(1),
                        noiseSigma(0.1),
                        noisyTestData(false)
        {
        }

        arma::vec lengthScale;
        double signalSigma;
        double noiseSigma;
        bool noisyTestData;
    };

    enum CovFunctionLabel
    {
        rbf = 0
    };

public:
    /*!
     * Constructor.
     * \param phi features of the regressor
     * \param covFunction label to specify which covariance function to use
     */
    GaussianProcess_(Features_<InputC, denseOutput>& phi, CovFunctionLabel covFunction = rbf) :
        BatchRegressor_<InputC, arma::vec, denseOutput>(phi),
        covFunction(covFunction)
    {
        hParams.lengthScale = arma::vec(phi.rows(), arma::fill::ones) * hParams.lengthScale(0);
    }

    /*!
     * Compute the mean of the function in the given point.
     * \param testInput vector of input
     * \return the mean
     */
    virtual arma::vec operator()(const InputC& testInputs) override
    {
        arma::mat testFeatures = this->phi(testInputs);
        arma::vec outputs(1, arma::fill::zeros);

        arma::vec k = generateCovVector(testFeatures.col(0));
        outputs(0) = arma::dot(k.t(), alpha);

        return outputs;
    }

    /*!
     * Compute the variance of the function in the given point.
     * \param testInput vector of input
     * \return the variance
     */
    double computeVariance(const InputC& testInputs)
    {
        arma::mat testFeatures = this->phi(testInputs);

        arma::vec k = generateCovVector(testFeatures.col(0));
        arma::vec v = arma::solve(arma::trimatl(L), k, arma::solve_opts::fast);
        double var = computeKernel(testFeatures.col(0), testFeatures.col(0)) -
                     arma::dot(v.t(), v);

        if(hParams.noisyTestData)
            var += hParams.noiseSigma * hParams.noiseSigma;

        return var;
    }

    /*!
     * Train the Gaussian Process with a given dataset.
     * \param featureDataset the dataset to be used to train the model
     */
    virtual void trainFeatures(const BatchData_<arma::vec, denseOutput>& featureDataset) override
    {
        features = featureDataset.getFeatures();
        typename output_traits<arma::vec>::type out = featureDataset.getOutputs();
        outputs = out.row(0).t();

        arma::mat K = generateCovMatrix(features);

        L = arma::chol(K, "lower");
        alpha = arma::solve(arma::trimatu(L.t()),
                            arma::solve(arma::trimatl(L),
                                        outputs,
                                        arma::solve_opts::fast),
                            arma::solve_opts::fast);
    }

    void updateAlpha(const arma::vec& outputs)
    {
        this->outputs = outputs;

        alpha = arma::solve(arma::trimatu(L.t()),
                            arma::solve(arma::trimatl(L),
                                        this->outputs,
                                        arma::solve_opts::fast),
                            arma::solve_opts::fast);
    }

    /*!
     * Compute the mean squared error between the mean of the Gaussian Process in the
     * given point and the ground truth value.
     * \param featureDataset the dataset to be used to test the model
     * \return the mean squared error
     */
    virtual double computeJFeatures(const BatchData_<arma::vec, denseOutput>& featureDataset) override
    {
        features = featureDataset.getFeatures();
        typename output_traits<arma::vec>::type out = featureDataset.getOutputs();

        double J = 0;
        unsigned int nSamples = features.n_cols;
        auto& self = *this;
        arma::vec y = out.row(0).t();

        for(unsigned int i = 0; i < nSamples; i++)
        {
            arma::vec yHat = self(features.col(i));

            J += arma::norm(y.row(i).t() - yHat(0));
        }

        J /= 2 * nSamples;

        return J;
    }

    /*!
     * Compute the marginal likelihood of the Gaussian Process.
     * \return the marginal likelihood
     */
    double computeMarginalLikelihood()
    {
        return -0.5 * arma::dot(outputs.t(), alpha) - arma::sum(log(L.diag())) -
               L.n_rows * 0.5 * log(2 * M_PI);
    }

    /*!
     * Setter.
     * \param alpha vector of weights
     */
    void setAlpha(const arma::vec& alpha)
    {
        this->alpha = alpha;
    }

    /*!
     * Setter.
     * \param inputs training set to be set
     */
    void setFeatures(const typename input_traits<denseOutput>::type inputs)
    {
    	typename input_traits<denseOutput>::type features(this->phi.rows(), inputs.n_rows, arma::fill::zeros);
        for(int i = 0; i < inputs.n_rows; i++)
            features.col(i) = this->phi(inputs.row(i).t());
        this->features = features;
    }

    /*!
     * Getter.
     * \return vector of weights
     */
    const arma::vec& getAlpha() const
    {
    	return alpha;
    }

    /*!
     * Getter.
     * \return training set
     */
    const typename input_traits<denseOutput>::type& getFeatures() const
    {
    	return features;
    }

    /*!
     * Getter.
     * \return the hyperparameters of the model
     */
    HyperParameters& getHyperParameters()
    {
        return hParams;
    }

    virtual ~GaussianProcess_()
    {
    }

protected:
    arma::mat L;
    arma::vec alpha;
    typename input_traits<denseOutput>::type features;
    arma::vec outputs;
    HyperParameters hParams;
    CovFunctionLabel covFunction;

protected:
    arma::mat generateCovMatrix(typename input_traits<denseOutput>::type testFeatures)
    {
        arma::mat K(testFeatures.n_cols, testFeatures.n_cols, arma::fill::zeros);

        for(unsigned int i = 0; i < K.n_rows; i++)
            for(unsigned int j = i; j < K.n_cols; j++)
                K(i, j) = K(j, i) = computeKernel(testFeatures.col(i), testFeatures.col(j), i == j);

        return K;
    }

    arma::vec generateCovVector(arma::vec testFeatures)
    {
        arma::vec k(features.n_cols, arma::fill::zeros);

        for(unsigned int i = 0; i < k.n_elem; i++)
            k(i) = computeKernel(testFeatures, features.col(i));

        return k;
    }

    double computeKernel(arma::vec x_p, arma::vec x_q, bool sameIndex = false)
    {
        double k = 0;
        arma::mat M(x_p.n_elem, x_p.n_elem, arma::fill::zeros);
        M.diag() = 1 / (hParams.lengthScale % hParams.lengthScale);

        if(covFunction == rbf)
        {
            k = hParams.signalSigma * hParams.signalSigma *
                exp(- 0.5 * arma::dot((x_p - x_q).t(), M * (x_p - x_q)));
            if(sameIndex)
                k += hParams.noiseSigma * hParams.noiseSigma;
        }
        else
        {

        }

        return k;
    }
};

typedef GaussianProcess_<arma::vec> GaussianProcess;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_OTHERS_GAUSSIANPROCESS_H_ */
