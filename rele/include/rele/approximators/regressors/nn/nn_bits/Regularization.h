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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_BITS_REGULARIZATION_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_BITS_REGULARIZATION_H_

#include <armadillo>

class Regularization
{
public:
    virtual double cost(const arma::vec& w) = 0;
    virtual arma::vec diff(const arma::vec& w) = 0;

    virtual ~Regularization()
    {

    }
};

class NoRegularization : public Regularization
{
public:
    inline virtual double cost(const arma::vec& w) override
    {
        return 0;
    }

    inline virtual arma::vec diff(const arma::vec& w) override
    {
        return arma::vec(w.n_elem, arma::fill::zeros);
    }
};

class L2_Regularization : public Regularization
{
public:
    inline virtual double cost(const arma::vec& w) override
    {
        return arma::norm(w);
    }

    inline virtual arma::vec diff(const arma::vec& w) override
    {
        return 2*w;
    }


};


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_BITS_REGULARIZATION_H_ */
