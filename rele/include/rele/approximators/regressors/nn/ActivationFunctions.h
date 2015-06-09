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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_ACTIVATIONFUNCTIONS_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_ACTIVATIONFUNCTIONS_H_

#include <cmath>

class Function
{
public:
    virtual double operator() (double x) = 0;
    virtual double diff(double x) = 0;

    virtual ~Function()
    {

    }
};

class Sigmoid : public Function
{
public:
    inline virtual double operator() (double x)
    {
        return 1.0 / ( 1.0 + std::exp(-x));
    }

    inline virtual double diff(double x)
    {
        Sigmoid& f = *this;
        return f(1 - f(x));
    }
};

class HyperbolicTangent : public Function
{
public:
    inline virtual double operator() (double x)
    {
        return std::tanh(x);
    }

    inline virtual double diff(double x)
    {
        return 1 - std::pow(std::tanh(x), 2);
    }
};

class Rectifier : public Function
{
public:
    inline virtual double operator() (double x)
    {
        return std::log(1 - std::exp(x));
    }

    inline virtual double diff(double x)
    {
        return 1.0 / ( 1.0 + std::exp(-x));
    }
};


class Linear : public Function
{
public:
    Linear(double alpha = 1) : alpha(alpha)
    {

    }

    inline virtual double operator() (double x)
    {
        return alpha*x;
    }

    inline virtual double diff(double x)
    {
        return alpha;
    }

private:
    double alpha;
};

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_ACTIVATIONFUNCTIONS_H_ */
