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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_BITS_ACTIVATIONFUNCTIONS_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_BITS_ACTIVATIONFUNCTIONS_H_

#include <cmath>

class Function
{
public:
    inline arma::vec operator()(const arma::vec& x)
    {
        arma::vec fx = x;

        auto fun = [&](double val)
        {
            return this->eval(val);
        };

        fx.transform(fun);

        return fx;

    }

    inline arma::vec diff(const arma::vec& x)
    {
        arma::vec dfx = x;

        auto fun = [&](double val)
        {
            return this->diff(val);
        };

        dfx.transform(fun);

        return dfx;

    }

    virtual double eval(double x) = 0;
    virtual double diff(double x) = 0;

    virtual ~Function()
    {

    }
};

class Sigmoid: public Function
{
public:
    inline virtual double eval(double x) override
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline virtual double diff(double x) override
    {
        return eval(x) * (1 - eval(x));
    }
};

class HyperbolicTangent: public Function
{
public:
    inline virtual double eval(double x) override
    {
        return std::tanh(x);
    }

    inline virtual double diff(double x) override
    {
        return 1 - std::pow(std::tanh(x), 2);
    }
};

class SoftPlus: public Function
{
public:
    inline virtual double eval(double x) override
    {
        return std::log(1 + std::exp(x));
    }

    inline virtual double diff(double x) override
    {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

class ReLU: public Function
{
public:
    inline virtual double eval(double x) override
    {
        return std::max(0.0, x);
    }

    inline virtual double diff(double x) override
    {
        return (x > 0) ? 1.0 : 0.0;
    }
};

class Linear: public Function
{
public:
    Linear(double alpha = 1) :
        alpha(alpha)
    {

    }

    inline virtual double eval(double x) override
    {
        return alpha * x;
    }

    inline virtual double diff(double x) override
    {
        return alpha;
    }

private:
    double alpha;
};

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_BITS_ACTIVATIONFUNCTIONS_H_ */
