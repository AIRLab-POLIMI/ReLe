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

#ifndef INCLUDE_RELE_POLICY_Q_POLICY_BOLTZMANN_H_
#define INCLUDE_RELE_POLICY_Q_POLICY_BOLTZMANN_H_

#include "ActionValuePolicy.h"

namespace ReLe
{

class Boltzmann: public ActionValuePolicy<FiniteState>
{
public:
    Boltzmann();
    virtual ~Boltzmann();

    virtual unsigned int operator()(const size_t& state) override;
    virtual double operator()(const size_t& state, const unsigned int& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "Boltzmann";
    }

    virtual hyperparameters_map getPolicyHyperparameters() override;

    void setTemperature(double tau)
    {
        this->tau = tau;
    }

    double getTemperature()
    {
        return this->tau;
    }

    virtual Boltzmann* clone() override
    {
        return new Boltzmann(*this);
    }

protected:
    double tau;

private:
    arma::vec computeProbabilities(size_t state);
};

class BoltzmannApproximate: public ActionValuePolicy<DenseState>
{
public:
    BoltzmannApproximate();
    virtual ~BoltzmannApproximate();

    virtual unsigned int operator()(const arma::vec& state) override;
    virtual double operator()(const arma::vec& state, const unsigned int& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "Approximate Boltzmann";
    }
    virtual hyperparameters_map getPolicyHyperparameters() override;

    void setTemperature(double tau)
    {
        this->tau = tau;
    }

    double getTemperature()
    {
        return this->tau;
    }

    virtual BoltzmannApproximate* clone() override
    {
        return new BoltzmannApproximate(*this);
    }

private:
    arma::vec computeProbabilities(const arma::vec& state);

protected:
    double tau;

};

}

#endif /* INCLUDE_RELE_POLICY_Q_POLICY_BOLTZMANN_H_ */
