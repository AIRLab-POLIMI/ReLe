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

#ifndef DISTRIBUTION_H_
#define DISTRIBUTION_H_

#include <armadillo>

namespace ReLe
{

/*!
 * This is an interface for a generic distribution.
 * A distribution is a multivariate statistical distribution that can change over time.
 * This interface implements the operators to get the pdf of the distribution at a certain point,
 * and to sample data according to the current state of the distribution.
 *
 * Distribution can be used for describing high level policies, i.e. a distribution of parametric policies.
 *
 */
class Distribution
{
public:

    /*!
     * Constructor.
     * \param dim the number of variables of the distribution
     */
    Distribution(unsigned int dim) : pointSize(dim)
    { }

    /*!
     * Destructor.
     */
    virtual ~Distribution()
    { }

    /*!
     * Draw a point from the support of the distribution
     * according to the probability defined by the distribution
     *
     * \return a randomly generated point
     */
    virtual arma::vec operator() () const = 0;

    /*!
     * Return the probability of a point to be generated
     * from the distribution.
     * \param point a point to be evaluated
     * \return the probability of the point
     */
    virtual double operator() (const arma::vec& point) const = 0;

    /*!
     * Return the logarithm of the probability of a point to
     * be generated from the distribution.
     * \param point a point to be evaluated
     * \return the logarithm of the probability of the point
     */
    virtual double logPdf(const arma::vec& point) const
    {
        auto& self = *this;
        return self(point);
    }

    /*!
     * Getter.
     * \return The size of the support
     */
    inline unsigned int getPointSize() const
    {
        return pointSize;
    }

    /*!
     * Getter.
     * \return the name of the distribution
     */
    virtual std::string getDistributionName() const = 0;

    /*!
     * Getter.
     * \return the distribution mean
     */
    virtual arma::mat getMean() const = 0;

    /*!
     * Getter.
     * \return the distribution covariance
     */
    virtual arma::mat getCovariance() const = 0;

    /*!
     * Getter.
     * \return the distribution mode
     */
    virtual arma::mat getMode() const = 0;

    /*!
     * This method implements the weighted maximum likelihood estimate of
     * the distribution, given a set of weighted samples.
     *
     * \param weights the weights for each sample.
     * \param samples the set of samples from the distribution
     */
    virtual void wmle(const arma::vec& weights, const arma::mat& samples) = 0;

protected:
    //! the number of variables of the distribution
    unsigned int pointSize;

};

/*!
 * This class represents a generic parametrized distribution \f$x \sim D(\cdot|\rho)\f$
 * where \f$\rho \in \mathbb{R}^{d}\f$ is the parameter vector and \f$ X \subseteq \mathbb{R}^n \f$ is
 * the support space.
 *
 */
class DifferentiableDistribution : public Distribution
{

public:
    /*!
     * Constructor.
     * \param dim the number of variables of the distribution
     */
    DifferentiableDistribution(unsigned int dim)
        : Distribution(dim)
    { }

    virtual ~DifferentiableDistribution()
    { }

    /*!
     * Getter.
     * \return The size of the parameters
     */
    virtual unsigned int getParametersSize() const = 0;

    /*!
     * Getter.
     * \return The parameters vector
     */
    virtual arma::vec getParameters() const = 0;

    /*!
     * Setter.
     * \param parameters The new parameters of the distribution.
     */
    virtual void setParameters(const arma::vec& parameters) = 0;

    /*!
     * Update the internal parameters according to the
     * given increment vector.
     *
     * \param increment a vector of increment value for each component
     */
    virtual void update(const arma::vec& increment) = 0;

    /*!
     * Compute the gradient of the logarithm of the distribution
     * in the given point
     * \param point the point where the gradient is evaluated
     * \return the gradient vector
     */
    virtual arma::vec difflog(const arma::vec& point) const = 0;

    /*!
     * Compute the hessian (\f$d(d\log D)^{T}\f$) of the logarithm of the
     * distribution in the given point.
     * \param point the point where the hessian is evaluated
     * \return the hessian matrix (out)
     */
    virtual arma::mat diff2log(const arma::vec& point) const = 0;

    /*!
     * Compute the gradient of the logarithm of the distribution
     * in the current params w.r.t. the given point.
     * differently from difflog, the computed gradient is not the parameter's
     * gradient, but the input gradient, i.e. how much the probability changes
     * if the input changes.
     * \param point the point where the gradient is evaluated
     * \return the gradient vector
     */
    virtual arma::vec pointDifflog(const arma::vec& point) const = 0;


};

/*!
 * This interface can be implemented from a distribution that has a closed form fisher
 * information matrix computation.
 */
class FisherInterface
{
public:
    /*!
     * Destructor
     */
    virtual ~FisherInterface()
    {
    }

    /*!
     * Computes the fisher information matrix of the distribution.
     */
    virtual arma::sp_mat FIM() const = 0;

    /*!
     * Computes the inverse of the fisher information matrix.
     */
    virtual arma::sp_mat inverseFIM() const = 0;
};

} //end namespace
#endif //DISTRIBUTION_H_
