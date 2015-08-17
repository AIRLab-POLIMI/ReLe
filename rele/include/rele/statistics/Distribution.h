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
#include "Interfaces.h"

namespace ReLe
{

class Distribution : public WritableInterface
{
public:

    Distribution(unsigned int dim) : pointSize(dim)
    { }

    virtual ~Distribution()
    { }

    /**
     * Draw a point from the support of the distribution
     * according to the probability defined by the distribution
     *
     * @brief Distribution sampling
     * @param output a randomly generated point
     */
    virtual arma::vec operator() () = 0;

    /**
     * Return the probability of a point to be generated
     * from the distribution.
     * @brief Distribution probability
     * @param point a point to be evaluated
     * @return the probability of the point
     */
    virtual double operator() (arma::vec& point) = 0;

    /**
     * @brief Support dimension
     * @return The size of the support
     */
    inline unsigned int getPointSize()
    {
        return pointSize;
    }

    virtual std::string getDistributionName() = 0;

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) = 0;

protected:
    unsigned int pointSize;

};

/**
 * This class represents a generic parametrized distribution \f$x \sim D(\cdot|\rho)\f$
 * where \f$\rho \in R^{d}\f$ is the parameter vector and \f$ X \subseteq R^n \f$ is
 * the support space.
 *
 * @brief A parametric statistic distribution
 */
class DifferentiableDistribution : public Distribution
{

public:

    DifferentiableDistribution(unsigned int support_size)
        : Distribution(support_size)
    { }

    virtual ~DifferentiableDistribution()
    { }

    /**
     * @brief Parameters size
     * @return The size of the parameters
     */
    virtual unsigned int getParametersSize() = 0;

    virtual arma::vec getParameters() = 0;

    /**
     * Update the internal parameters according to the
     * given increment vector.
     *
     * @brief Update the parameters
     * @param increment a vector of increment value for each component
     */
    virtual void update(arma::vec& increment) = 0;

    /**
     * Compute the gradient of the logarithm of the distribution
     * in the given point
     * @brief Log-gradient computation
     * @param point the point where the gradient is evaluated
     * @param gradient The gradient vector (out)
     */
    virtual arma::vec difflog(const arma::vec& point) = 0;


    /**
     * Compute the hessian (\f[d (d \log D)^{T}\f]) of the logarithm of the
     * distribution in the given point.
     * @brief Log-hessian computation
     * @param point the point where the gradient is evaluated
     * @param hessian The hessian matrix (out)
     */
    virtual arma::mat diff2log(const arma::vec& point) = 0;

};


class FisherInterface
{
public:
    virtual ~FisherInterface()
    {
    }

    virtual arma::sp_mat FIM() = 0;
    virtual arma::sp_mat inverseFIM() = 0;
};

} //end namespace
#endif //DISTRIBUTION_H_
