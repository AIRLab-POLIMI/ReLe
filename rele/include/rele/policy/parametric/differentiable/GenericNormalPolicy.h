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

#ifndef INCLUDE_RELE_POLICY_PARAMETRIC_DIFFERENTIABLE_GENERICNORMALPOLICY_H_
#define INCLUDE_RELE_POLICY_PARAMETRIC_DIFFERENTIABLE_GENERICNORMALPOLICY_H_

#include "Policy.h"
#include "Regressors.h"
#include "ArmadilloPDFs.h"

#include <cassert>

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY
///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Multivariate Normal policy with fixed covariance matrix
 *
 * This class represents a multivariate Normal policy with fixed covariance matrix
 * and generic approximation of the mean value:
 * \f[ \pi^{\theta} (a|s) = \mathcal{N}(s; \mu(\phi(s),\theta), \Sigma),\qquad
 * \forall s \in R^{n_s}, a \in R^{n_a},\f]
 * where \f$\phi(s)\f$ is an \f$(n_a \times k)\f$ matrix and
 * \f$\theta\f$ is a \f$k\f$-dimensional vector.
 */
class GenericMVNPolicy: public DifferentiablePolicy<DenseAction, DenseState>
{
public:

    /**
     * Create an instance of the class using the given projector.
     * Covariance matrix is initialized to the unit matrix.
     * Note that the weights of the mean approximation are not
     * changed, i.e., the initial weights are specified by the
     * instance of the regressor received as parameter.
     *
     * @brief The constructor.
     * @param approximator The regressor used for mean approximation
     */
    GenericMVNPolicy(ParametricRegressor& approximator) :
        meanApproximator(approximator),
        mean(approximator.getOutputSize(), arma::fill::zeros)
    {
        int output_dim = approximator.getOutputSize();
        Sigma.eye(output_dim, output_dim);
        invSigma = arma::inv(Sigma);
        choleskySigma = arma::chol(Sigma);
        determinant = arma::det(Sigma);
    }

    GenericMVNPolicy(ParametricRegressor& approximator, arma::mat& covariance) :
        meanApproximator(approximator),
        mean(approximator.getOutputSize(), arma::fill::zeros),
        Sigma(covariance)
    {
        invSigma = arma::inv(Sigma);
        choleskySigma = arma::chol(Sigma);
        determinant = arma::det(Sigma);
    }

    /**
     * Create an instance of the class using the given projector and
     * covariance matrix.
     *
     * @brief The constructor.
     * @param approximator The regressor used for mean approximation.
     * @param initialCov The covariance matrix (\f$n_a \times n_a\f$).
     */
    GenericMVNPolicy(ParametricRegressor& approximator,
                     std::initializer_list<double> initialCov) :
        meanApproximator(approximator),
        mean(approximator.getOutputSize(), arma::fill::zeros)
    {
        int output_dim = approximator.getOutputSize();
        Sigma.zeros(output_dim, output_dim);
        int row = 0, col = 0;
        for (double x : initialCov)
        {
            Sigma(row, col++) = x;
            if (col == output_dim)
            {
                col = 0;
                ++row;
            }
        }
        invSigma = arma::inv(Sigma);
        choleskySigma = arma::chol(Sigma);
        determinant = arma::det(Sigma);
    }

    GenericMVNPolicy(ParametricRegressor& approximator, double* covariance) :
        meanApproximator(approximator),
        mean(approximator.getOutputSize(), arma::fill::zeros)
    {
        int output_dim = approximator.getOutputSize();
        Sigma.zeros(output_dim, output_dim);
        for (int i = 0; i < output_dim; ++i)
        {
            for (int j = 0; j < output_dim; ++j)
            {
                Sigma(i, j) = covariance[i + output_dim * j];
            }
        }

        invSigma = arma::inv(Sigma);
        choleskySigma = arma::chol(Sigma);
        determinant = arma::det(Sigma);
    }

    virtual ~GenericMVNPolicy()
    {

    }

    virtual inline std::string getPolicyName() override
    {
        return "MVNPolicy";
    }
    virtual inline std::string getPolicyHyperparameters() override
    {
        return "";
    }
    virtual inline std::string printPolicy() override
    {
        return "";
    }

public:

    virtual double operator()(const arma::vec& state, const arma::vec& action) override;

    virtual arma::vec operator()(const arma::vec& state) override;

    virtual GenericMVNPolicy* clone() override
    {
        return new  GenericMVNPolicy(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return meanApproximator.getParameters();
    }
    virtual inline const unsigned int getParametersSize() const override
    {
        return meanApproximator.getParametersSize();
    }
    virtual inline void setParameters(const arma::vec& w) override
    {
        meanApproximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(const arma::vec& state, const arma::vec& action) override;
    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;

protected:

    /**
     * This function is deputed to the computatio of the mean and covariance
     * values in the given state. Moreover, the function must compute all the
     * informations required for the generation of samples from the Gaussian
     * distribution and gradient computation. In particular, the flag \a cholesky_dec
     * is used to require the computation of the cholesky decomposition of the
     * covariance matrix.
     *
     * In this base version only the mean value is updated since the covariance
     * matrix is indipendent from the state value.
     *
     * @brief Update internal state.
     * @param state The state where the policy is evaluated.
     * @param cholesky_dec A flag used to require the Cholesky decomposition of the
     * covariance matrix.
     */
    inline virtual void updateInternalState(const arma::vec& state, bool cholesky_dec = false)
    {
        // compute mean vector
        mean = meanApproximator(state);
    }

protected:
    arma::mat Sigma, invSigma, choleskySigma;
    double determinant;
    ParametricRegressor& meanApproximator;
    arma::vec mean;
};


///////////////////////////////////////////////////////////////////////////////////////
/// Generic MVN POLICY with Diagonal covariance (parameters: mean, diagonal standard deviations)
///////////////////////////////////////////////////////////////////////////////////////
class GenericMVNDiagonalPolicy : public GenericMVNPolicy
{
public:
    GenericMVNDiagonalPolicy(ParametricRegressor& approximator)
        : GenericMVNPolicy(approximator), stddevParams(approximator.getOutputSize(), arma::fill::ones)
    {
        UpdateCovarianceMatrix();
    }

    GenericMVNDiagonalPolicy(ParametricRegressor& approximator,
                             arma::vec stddevVector)
        : GenericMVNPolicy(approximator), stddevParams(stddevVector)
    {
        UpdateCovarianceMatrix();
    }

    virtual ~GenericMVNDiagonalPolicy()
    {

    }

    virtual inline std::string getPolicyName() override
    {
        return "GenericMVNDiagonalPolicy";
    }
    virtual inline std::string getPolicyHyperparameters() override
    {
        return "";
    }
    virtual inline std::string printPolicy() override
    {
        return "";
    }

    virtual GenericMVNDiagonalPolicy* clone() override
    {
        return new  GenericMVNDiagonalPolicy(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return vectorize(meanApproximator.getParameters(), stddevParams);
    }
    virtual inline const unsigned int getParametersSize() const override
    {
        return meanApproximator.getParametersSize() + stddevParams.n_elem;
    }
    virtual void setParameters(const arma::vec& w) override;

    // DifferentiablePolicy interface
public:

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;

private:
    /**
     * Compute the covariance matrix from the logistic parameters and
     * compute the Cholesky decomposition of it.
     *
     * @brief Update the covariance matrix
     */
    void UpdateCovarianceMatrix();

protected:
    arma::vec stddevParams;
};


///////////////////////////////////////////////////////////////////////////////////////
/// Generic MVN POLICY with state dependant diagonal standard deviation (parameters: mean, std dev weights)
///////////////////////////////////////////////////////////////////////////////////////
class GenericMVNStateDependantStddevPolicy : public GenericMVNPolicy
{
public:
    GenericMVNStateDependantStddevPolicy(ParametricRegressor& meanApproximator, ParametricRegressor& stdApproximator)
        : GenericMVNPolicy(meanApproximator), stdApproximator(stdApproximator)
    {
        assert(meanApproximator.getOutputSize() == stdApproximator.getOutputSize());

        // initialize covariance matrix and inverse
        int n = stdApproximator.getOutputSize();
        Sigma.zeros(n, n);
        invSigma.zeros(n, n);
        choleskySigma.zeros(n, n);
    }

    virtual inline std::string getPolicyName() override
    {
        return "GenericMVNStateDependantStddevPolicy";
    }
    virtual inline std::string getPolicyHyperparameters() override
    {
        return "";
    }
    virtual inline std::string printPolicy() override
    {
        return "";
    }

    virtual GenericMVNStateDependantStddevPolicy* clone() override
    {
        return new  GenericMVNStateDependantStddevPolicy(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return vectorize(meanApproximator.getParameters(), stdApproximator.getParameters());
    }

    virtual inline const unsigned int getParametersSize() const override
    {
        return meanApproximator.getParametersSize() + stdApproximator.getParametersSize();
    }

    virtual inline void setParameters(const arma::vec& w) override
    {
        assert(w.n_elem == this->getParametersSize());
        meanApproximator.setParameters( w.rows(0,meanApproximator.getParametersSize() - 1) );
        stdApproximator.setParameters( w.rows(meanApproximator.getParametersSize(), w.n_elem-1) );
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;

protected:
    virtual void updateInternalState(const arma::vec& state, bool cholesky_dec = false) override;


protected:
    ParametricRegressor& stdApproximator;

};



}



#endif /* INCLUDE_RELE_POLICY_PARAMETRIC_DIFFERENTIABLE_GENERICNORMALPOLICY_H_ */
