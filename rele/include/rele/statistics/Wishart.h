/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_STATISTICS_WISHART_H_
#define INCLUDE_RELE_STATISTICS_WISHART_H_

#include "rele/statistics/Distribution.h"

namespace ReLe
{

/*!
 * This class is the base class for Wishart and Inverse-Wishart
 * Distributions.
 */
class WishartBase : public Distribution
{
public:
    /*!
     * Constructor.
     * \param p the number of columns (and rows) of the sampled matrix
     */
    WishartBase(unsigned int p);

    /*!
     * Constructor.
     * \param p the number of columns (and rows) of the sampled matrix
     * \param nu the degrees of freedom of the distribution
     */
    WishartBase(unsigned int p, unsigned int nu);

    /*!
     * Getter.
     * \return the degrees of freedom of the distribution
     */
    inline unsigned int getNu() const
    {
        return nu;
    }


    virtual double operator() (const arma::vec& point) const override;
    virtual double logPdf(const arma::vec& point) const override = 0;


protected:
    double lgamma_p(unsigned int p, double value) const;

protected:
    unsigned int nu;

};

/*!
 * This class implements a Wishart distribution.
 * This distribution is commonly used for precision matrix estimation.
 */
class Wishart : public WishartBase
{
public:
    /*!
     * Constructor.
     * \param p the number of rows and columns in the matrix
     */
    Wishart(unsigned int p);

    /*!
     * Constructor.
     * \param p the number of rows and columns in the matrix
     * \param nu he degrees of freedom of the wishart distribution
     */
    Wishart(unsigned int p, unsigned int nu);

    /*!
     * Constructor.
     * \param nu the degrees of freedom of the wishart distribution
     * \param V the covariance of the distribution
     */
    Wishart(unsigned int nu,
            const arma::mat& V);

    /*!
     * Getter.
     * \return the covariance matrix of the distribution
     */
    inline arma::mat getV() const
    {
        return V;
    }

    /*!
     * Destructor.
     */
    virtual ~Wishart();

    // Distribution interface
public:
    virtual arma::vec operator() () const override;
    virtual double logPdf(const arma::vec& point) const override;

    virtual inline std::string getDistributionName() const override
    {
        return "InverseWishart";
    }

    inline arma::mat getMean() const override
    {

        return nu*V;
    }

    inline arma::mat getCovariance() const override
    {
        //TODO [IMPORTANT] implement
        return arma::mat();
    }

    inline arma::mat getMode() const override
    {
        unsigned int p = V.n_cols;
        return (nu - p - 1.0)*V;
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

private:
    arma::mat V;

};

/*!
 * This class implements a Wishart distribution.
 * This distribution is commonly used for precision matrix estimation.
 */
class InverseWishart : public WishartBase
{
public:
    /*!
     * Constructor.
     * \param p the number of rows and columns in the matrix
     */
    InverseWishart(unsigned int p);

    /*!
     * Constructor.
     * \param p the number of rows and columns in the matrix
     * \param nu he degrees of freedom of the wishart distribution
     */
    InverseWishart(unsigned int p, unsigned int nu);

    /*!
     * Constructor.
     * \param nu the degrees of freedom of the wishart distribution
     * \param Psi the scale matrix of the distribution
     */
    InverseWishart(unsigned int nu,
                   const arma::mat& Psi);

    /*!
     * Getter.
     * \return the scale matrix of the distribution
     */
    inline arma::mat getPsi() const
    {
        return Psi;
    }

    /*!
     * Destructor.
     */
    virtual ~InverseWishart();

    // Distribution interface
public:
    virtual arma::vec operator() () const override;
    virtual double logPdf(const arma::vec& point) const override;

    virtual inline std::string getDistributionName() const override
    {
        return "Wishart";
    }

    inline arma::mat getMean() const override
    {
        unsigned int p = Psi.n_cols;
        return Psi/(nu - p - 1.0);
    }

    inline arma::mat getCovariance() const override
    {
        //TODO [IMPORTANT] implement
        return arma::mat();
    }

    inline arma::mat getMode() const override
    {
        unsigned int p = Psi.n_cols;
        return Psi/(nu + p + 1.0);
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

private:
    arma::mat Psi;

};

}

#endif /* INCLUDE_RELE_STATISTICS_WISHART_H_ */
