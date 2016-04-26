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

#ifndef E_GREEDY_H_
#define E_GREEDY_H_

#include "rele/policy/q_policy/ActionValuePolicy.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"

namespace ReLe
{

class e_Greedy: public ActionValuePolicy<FiniteState>
{
public:
    e_Greedy();
    virtual ~e_Greedy();

    virtual unsigned int operator()(const size_t& state) override;
    virtual double operator()(const size_t& state, const unsigned int& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "e-Greedy";
    }
    virtual hyperparameters_map getPolicyHyperparameters() override;

    inline void setEpsilon(double eps)
    {
        this->eps = eps;
    }

    inline double getEpsilon()
    {
        return this->eps;
    }

    virtual e_Greedy* clone() override
    {
        return new e_Greedy(*this);
    }

protected:
    double eps;
};

class e_GreedyApproximate: public ActionValuePolicy<DenseState>
{
public:
    e_GreedyApproximate();
    virtual ~e_GreedyApproximate();

    virtual unsigned int operator()(const arma::vec& state) override;
    virtual double operator()(const arma::vec& state, const unsigned int& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "Approximate e-Greedy";
    }
    virtual hyperparameters_map getPolicyHyperparameters() override;

    inline void setEpsilon(double eps)
    {
        this->eps = eps;
    }

    inline double getEpsilon()
    {
        return this->eps;
    }

    virtual e_GreedyApproximate* clone() override
    {
        return new e_GreedyApproximate(*this);
    }

protected:
    double eps;

};

class e_GreedyMultipleRegressors: public ActionValuePolicy<DenseState>
{
public:
    e_GreedyMultipleRegressors(std::vector<std::vector<GaussianProcess*>>& regressors);
    virtual ~e_GreedyMultipleRegressors();

    virtual unsigned int operator()(const arma::vec& state) override;
    virtual double operator()(const arma::vec& state, const unsigned int& action) override;

    virtual hyperparameters_map getPolicyHyperparameters() override;

    inline void setEpsilon(double eps)
    {
        this->eps = eps;
    }

    inline virtual std::string getPolicyName() override
    {
        return "MultipleRegressors e-Greedy";
    }

    inline void setRegressor(std::vector<std::vector<GaussianProcess*>>& regressors)
    {
        this->regressors = regressors;
    }

    inline double getEpsilon()
    {
        return this->eps;
    }

    virtual e_GreedyMultipleRegressors* clone() override
    {
        return new e_GreedyMultipleRegressors(*this);
    }

protected:
    double eps;
    std::vector<std::vector<GaussianProcess*>>& regressors;
};

}

#endif /* E_GREEDY_H_ */
