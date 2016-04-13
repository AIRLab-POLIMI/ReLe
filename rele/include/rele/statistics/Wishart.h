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

class Wishart : public Distribution
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
     * Destructor.
     */
    virtual ~Wishart();

    // Distribution interface
public:
    virtual arma::vec operator() () const override;
    virtual double operator() (const arma::vec& point) const override;

    virtual inline std::string getDistributionName() const override
    {
        return "Wishart";
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

protected:
    double tgamma_p(unsigned int p, double value) const;

private:
    arma::mat V;
    unsigned int nu;

};

}

#endif /* INCLUDE_RELE_STATISTICS_WISHART_H_ */
