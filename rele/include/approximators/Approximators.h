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
    virtual void evaluate (const DenseArray& input, DenseArray& output) = 0; //TODO anche questo con un operatore???
};

class ParametricRegressor : public Regressor
{

};

class NonParametricRegressor : public Regressor
{

};

class BatchRegressor
{

public:
	void train(/* classe che rappresenta il dataset da decidere*/);
};

class BasisFunction
{
public:
    virtual double operator() (const DenseArray& input) = 0; //Questo è una figata
    /**
     * @brief Write a complete description of the instance to
     * a stream.
     * @param out the output stream
     */
    virtual void WriteOnStream (std::ostream& out) = 0;

    /**
     * @brief Read the description of the basis function from
     * a file and reset the internal state according to that.
     * This function is complementary to WriteOnStream
     * @param in the input stream
     */
    virtual void ReadFromStream (std::istream& in) = 0;
};

class BasisFunctions : public std::vector<BasisFunction>
{
public:
    virtual void operator() (const DenseArray& input, DenseArray& output) = 0; //TODO NON sono tanto convinto di questo...
};


}


#endif /* APPROXIMATORS_H_ */
