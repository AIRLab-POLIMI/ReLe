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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSOR_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSOR_H_

namespace ReLe
{

#include <armadillo>

class QRegressor
{
public:
    virtual double operator()(const arma::vec& state, unsigned int action) = 0;
    virtual ~QRegressor()
    {

    }

};

class BatchQRegressor : public QRegressor
{
public:
    virtual void trainFeatures() = 0;

    virtual ~BatchQRegressor()
    {

    }

};

class ParametricQRegressor : public QRegressor
{
public:
    virtual void set(unsigned int action, const arma::vec& w) = 0;
    virtual void update(unsigned int action, const arma::vec& dw) = 0;
    virtual void diff(const arma::vec state, unsigned int action) = 0;

    virtual ~ParametricQRegressor()
    {

    }

};

}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSOR_H_ */
