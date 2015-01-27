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

#ifndef APPROXIMATORS_H_
#define APPROXIMATORS_H_

#include "Basics.h"

//TODO densearray deve essere una classe proxy... così puoi usare il container che vuoi...
//oppue deve avere una sottoclasse tipo DenseVector che abbia un vector all'internno...
namespace ReLe
{

class Regressor
{

public:

    Regressor(unsigned int input = 1, unsigned int output = 1) :
        inputDimension(input), outputDimension(output)
    {
    }

    virtual arma::vec operator() (const arma::vec& input) = 0;

    virtual ~Regressor()
    {
    }

protected:
    unsigned int inputDimension, outputDimension;
};

class ParametricRegressor: public Regressor
{
public:
    ParametricRegressor(unsigned int input = 1, unsigned int output = 1) :
        Regressor(input, output)
    {
    }

    virtual arma::vec& getParameters() = 0;
    virtual arma::vec  diff(const arma::vec& output) = 0;
};

class NonParametricRegressor: public Regressor
{
public:
    NonParametricRegressor(unsigned int input = 1, unsigned int output = 1) :
        Regressor(input, output)
    {
    }
};

class BatchRegressor
{

public:
    void train(/* classe che rappresenta il dataset da decidere*/);
};

}

#endif /* APPROXIMATORS_H_ */
