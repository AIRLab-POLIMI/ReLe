/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_POLICY_Q_POLICY_BOLTZMANN_H_
#define INCLUDE_RELE_POLICY_Q_POLICY_BOLTZMANN_H_

#include "ActionValuePolicy.h"

namespace ReLe
{

class Boltzmann: public ActionValuePolicy<FiniteState>
{
public:
	Boltzmann(arma::mat* Q);
    virtual ~Boltzmann();

    virtual int operator() (int state);
    virtual double operator() (int state, int action);

    void setEpsilon(double eps)
    {
        this->eps = eps;
    }

    double getEpsilon()
    {
        return this->eps;
    }

protected:
    arma::mat* Q;
    double eps;
};

class BoltzmannApproximate: public ActionValuePolicy<DenseState>
{
public:
	BoltzmannApproximate(Regressor* Q);
    virtual ~BoltzmannApproximate();

    virtual int operator() (const arma::vec& state);
    virtual double operator() (const arma::vec& state, int action);

    void setEpsilon(double eps)
    {
        this->eps = eps;
    }

    double getEpsilon()
    {
        return this->eps;
    }

protected:
    Regressor* Q;
    double eps;
    unsigned int nactions;
};

}

#endif /* INCLUDE_RELE_POLICY_Q_POLICY_BOLTZMANN_H_ */
