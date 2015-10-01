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

#ifndef SRC_TEST_ROCKY_ROCKYPOLICY_H_
#define SRC_TEST_ROCKY_ROCKYPOLICY_H_

#include "Policy.h"

namespace ReLe
{

class RockyPolicy : public ParametricPolicy<DenseAction, DenseState>
{
public:

    RockyPolicy(double dt);

    //Policy
    virtual arma::vec operator() (const arma::vec& state) override;
    virtual double operator() (const arma::vec& state, const arma::vec& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "Rocky Policy";
    }

    inline virtual std::string getPolicyHyperparameters() override
    {
        return "";
    }

    inline virtual std::string printPolicy() override
    {
        return "";
    }

    //ParametricPolicy
    inline virtual arma::vec getParameters() const override
    {
        return w;
    }

    inline virtual const unsigned int getParametersSize() const override
    {
        return PARAM_SIZE;
    }

    virtual void setParameters(const arma::vec& w) override
    {
        this->w = w;
    }

    virtual RockyPolicy* clone() override
    {
        return new RockyPolicy(*this);
    }


private:
    enum Objective
    {
        escape, feed, eat, home
    };

    enum StateComponents
    {
        //robot state
        x = 0,
        y,
        theta,

        //robot sensors
        energy,
        food,

        //rocky state
        xr,
        yr,
        thetar
    };

    enum ParameterComponents
    {
        escapeThreshold = 0,
        energyThreshold = 1,
        escapeParamsStart = 2,
        escapeParamsEnd = 13,
        PARAM_SIZE = 14
    };


private:
    Objective computeObjective(const arma::vec& state);

    arma::vec eatPolicy();
    arma::vec homePolicy(const arma::vec& state);
    arma::vec feedPolicy(const arma::vec& state);
    arma::vec escapePolicy(const arma::vec& state);

    arma::vec wayPointPolicy(const arma::vec& state, double ox, double oy);


private:
    arma::vec w;
    double dt;

private:
    const double maxV;
};

}


#endif /* SRC_TEST_ROCKY_ROCKYPOLICY_H_ */
