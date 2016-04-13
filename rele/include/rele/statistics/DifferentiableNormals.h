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

#ifndef DIFFERENTIABLENORMALS_H_
#define DIFFERENTIABLENORMALS_H_

#include "rele/statistics/Distribution.h"

namespace ReLe
{


/*!
 * This class implements the Parametric Normal distribution.
 * \f[ x\in\mathbb{R}^n, x\sim\mathcal{N}(\mu, \Sigma) \f]
 *
 * This is the basic class of all normal distributions, by default only
 * the mean is parametrized. This mean that the covariance matrix \f$\Sigma\f$
 * is fixed.
 *
 */
class ParametricNormal : public DifferentiableDistribution
{
public:
    /*!
     * Constructor.
     * \param dim the number of variables of the distribution
     */
    ParametricNormal(unsigned int dim);

    /*!
     * Constructor.
     * \param params the parameters of the distribution
     * \param covariance the covariance matrix
     */
    ParametricNormal(const arma::vec& params,
                     const arma::mat& covariance);

    /*!
     * Destructor.
     */
    virtual ~ParametricNormal()
    { }

    // Distribution interface
public:
    virtual arma::vec operator() () const override;
    virtual double operator() (const arma::vec& point) const override;

    virtual inline std::string getDistributionName() const override
    {
        return "ParametricNormal";
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // DifferentiableDistribution interface
public:

    inline unsigned int getParametersSize() const override
    {
        return mean.n_elem;
    }

    inline virtual arma::vec getParameters() const override
    {
        return mean;
    }

    virtual void setParameters(const arma::vec& newval) override;
    virtual void update(const arma::vec &increment) override;
    virtual arma::vec difflog(const arma::vec& point) const override;
    virtual arma::mat diff2log(const arma::vec& point) const override;
    virtual arma::vec pointDifflog(const arma::vec& point) const override;

public:
    //TODO [SERIALIZATION] check implementation.
    virtual void writeOnStream(std::ostream &out);
    virtual void readFromStream(std::istream &in);


    // Specific Normal policy interface
public:
    inline arma::mat getMean() const override
    {
        return mean;
    }

    inline arma::mat getCovariance() const override
    {
        return Cov;
    }

    inline arma::mat getMode() const override
    {
        return getMean();
    }

protected:
    /*!
     * Compute mean, covariance, inverse covariance and determinant values
     * according to current parameterization.
     *
     * @brief Update internal state
     */
    virtual void updateInternalState();

protected:
    arma::vec mean;
    arma::mat Cov, invCov, cholCov;
    double detValue;

};

/*!
 * This class represents a parametric Gaussian distribution with parameters \f$\rho\f$:
 * \f[x \sim \mathcal{N}(\cdot|\rho).\f]
 * The parameter vector \f$\rho\f$ is then defined as follows:
 * \f[\rho = [M, \Sigma]^{T}\f]
 * where \f$M=[\mu_1,\dots,\mu_n]\f$, \f$\Sigma = diag(\sigma_1, \dots,\sigma_n)\f$ and
 * \f$ n \f$ is the support dimension. As a consequence, the parameter
 * dimension is \f$2\cdot n\f$.
 *
 * Given a parametrization \f$\rho\f$, the distribution is defined by the mean
 * vector \f$M\f$ and a diagonal covariance matrix \f$\Sigma\f$.
 */
class ParametricDiagonalNormal : public ParametricNormal, public FisherInterface
{
public:
    /*!
     * Constructor.
     * \param mean the initial value for the mean
     * \param covariance the initial covariance
     */
    ParametricDiagonalNormal(const arma::vec& mean, const arma::vec& covariance);

    /*!
     * Destructor.
     */
    virtual ~ParametricDiagonalNormal()
    {}

    virtual inline std::string getDistributionName() const override
    {
        return "ParametricDiagonalNormal";
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) const override;
    arma::mat diff2log(const arma::vec& point) const override;

    arma::sp_mat FIM() const override;
    arma::sp_mat inverseFIM() const override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() const override;
    virtual arma::vec getParameters() const override;
    virtual void setParameters(const arma::vec& newval) override;
    virtual void update(const arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

private:
    arma::vec diagStdDev;
};


/*!
 * This class represents a parametric Gaussian distribution with parameters \f$\rho\f$:
 * \f[x \sim \mathcal{N}(\cdot|\rho).\f]
 * The parameter vector \f$\rho\f$ is then defined as follows:
 * \f[\rho = [M, \Omega]^{T}\f]
 * where \f$M=[\mu_1,\dots,\mu_n]\f$, \f$\Omega = [\omega_1, \dots,\omega_n]\f$ and
 * \f$ n \f$ is the support dimension. As a consequence, the parameter
 * dimension is \f$2\cdot n\f$.
 *
 * Given a parametrization \f$\rho\f$, the distribution is defined by the mean
 * vector \f$M\f$ and a covariance matrix \f$\Sigma\f$.
 * In order to reduce the number of parameters, we discard
 * the cross-correlation terms in the covariance matrix: \f$ \Sigma = diag(\sigma_1,\dots,\sigma_n)\f$.
 * Moreover, in order to prevent the variance from becoming negative we exploit
 * the parametrization presented by Kimura and Kobayashi (1998),
 * where \f$\sigma_i\f$ is represented by a logistic function parameterized by \f$\omega_i\f$:
 * \f[\sigma_i = \frac{\tau}{1+e^{-\omega_i}}.\f]
 */
class ParametricLogisticNormal : public ParametricNormal
{


public:
    /*!
     * Constructor.
     * \param point_dim the number of variables of the distribution
     * \param variance_asymptote the asymptotic value for the variance of each variable
     */
    ParametricLogisticNormal(unsigned int point_dim,
                             double variance_asymptote);

    /*!
     * Constructor.
     * \param mean the initial mean value
     * \param logWeights the initial weights for the logistic function
     * \param variance_asymptote the asymptotic value for the variance of each variable
     */
    ParametricLogisticNormal(const arma::vec& mean, const arma::vec& logWeights,
                             double variance_asymptote);

    /*!
     * Constructor.
     * \param variance_asymptote a vector of the asymptotic values for the variance of each variable
     */
    ParametricLogisticNormal(const arma::vec& variance_asymptote);

    /*!
     * Constructor.
     * \param mean the initial mean value
     * \param logWeights the initial weights for the logistic function
     * \param variance_asymptote a vector of the asymptotic values for the variance of each variable
     */
    ParametricLogisticNormal(const arma::vec& mean, const arma::vec& logWeights,
                             const arma::vec& variance_asymptote);

    /*!
     * Destructor.
     */
    virtual ~ParametricLogisticNormal()
    {}

    virtual inline std::string getDistributionName() const override
    {
        return "ParametricLogisticNormal";
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) const override;
    arma::mat diff2log(const arma::vec& point) const override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() const override;
    virtual arma::vec getParameters() const override;
    virtual void setParameters(const arma::vec& newval) override;
    virtual void update(const arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

private:

    inline double logistic(double w, double asymptote)
    {
        return asymptote / (1.0 + exp(-w));
    }

protected:
    arma::vec asVariance; //asymptotic varianceprivate:
    arma::vec logisticWeights; //weights used for the logistic function


};

/*!
 * This class represents a parametric Gaussian distribution with parameters \f$\rho\f$:
 * \f[x \sim \mathcal{N}(\cdot|\rho).\f]
 * The parameter vector \f$\rho\f$ is then defined as follows:
 * \f[\rho = [M, \Omega]^{T}\f]
 * where \f$M=[\mu_1,\dots,\mu_n]\f$, \f$\Omega = [\omega_1, \dots,\omega_n]\f$ and
 * \f$ n \f$ is the support dimension. As a consequence, the parameter
 * dimension is \f$2\cdot n\f$.
 *
 * Given a parametrization \f$\rho\f$, the distribution is defined by the mean
 * vector \f$M\f$ and a covariance matrix \f$\Sigma\f$.
 * In order to reduce the number of parameters and prevent the matrix become not positive definite,
 * we parametrize the covariance matrix with the cholesky decomposition of the Covariance matrix, such that:
 * \f[\Sigma=triangular(\Omega)^{T}triangular(\Omega)\f]
 */
class ParametricCholeskyNormal : public ParametricNormal, public FisherInterface
{

public:
    /*!
     * Constructor.
     * \param initial_mean the initial mean parameters
     * \param initial_cholA the initial cholesky decomposition of the covariance matrix
     */
    ParametricCholeskyNormal(const arma::vec& initial_mean,
                             const arma::mat& initial_cholA);

    /*!
     * Destructor.
     */
    virtual ~ParametricCholeskyNormal()
    {}

    virtual inline std::string getDistributionName() const override
    {
        return "ParametricCholeskyNormal";
    }

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) const override;
    arma::mat diff2log(const arma::vec& point) const override;

    arma::sp_mat FIM() const override;
    arma::sp_mat inverseFIM() const override;

    inline arma::mat getCholeskyDec()
    {
        return cholCov;
    }

    inline void setCholeskyDec(arma::mat& A)
    {
        cholCov = A;
        updateInternalState();
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() const override;
    virtual arma::vec getParameters() const override;
    virtual void setParameters(const arma::vec& newval) override;
    virtual void update(const arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

};

/*!
 * This class represents a parametric Gaussian distribution.
 *
 * Differently from ReLe::ParametricCholeskyNormal, it uses a full parametrization of the
 * covariance matrix, so the algorithm itself needs to maintain the positive definiteness
 * of the parametrization.
 *
 * Usually this class is used when the algorithm provide a full estimation of the
 * covariance matrix or for weighted maximum likelihood.
 *
 */
class ParametricFullNormal : public ParametricNormal, public FisherInterface
{

public:
    /*!
     * Constructor.
     * \param initial_mean the initial mean parameters
     * \param initial_cov the initial covariance matrix
     */
    ParametricFullNormal(const arma::vec& initial_mean,
                         const arma::mat& initial_cov);

    /*!
     * Destructor.
     */
    virtual ~ParametricFullNormal()
    {}

    virtual inline std::string getDistributionName() const override
    {
        return "ParametricFullNormal";
    }

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) const override;
    arma::mat diff2log(const arma::vec& point) const override;

    arma::sp_mat FIM() const override;
    arma::sp_mat inverseFIM() const override;

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() const override;
    virtual arma::vec getParameters() const override;
    virtual void setParameters(const arma::vec& newval) override;
    virtual void update(const arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

};





} //end namespace

#endif //DIFFERENTIABLENORMALS_H_
