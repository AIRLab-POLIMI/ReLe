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

#include "basis/PolynomialFunction.h"

using namespace arma;

namespace ReLe
{

PolynomialFunction::PolynomialFunction(std::vector<unsigned int> dimension, std::vector<unsigned int> degree)
    : dimension(dimension), degree(degree)
{
}

PolynomialFunction::PolynomialFunction(unsigned int _dimension, unsigned int _degree)
    : dimension(_dimension)
{
    for (unsigned i = 0; i < _dimension; ++i)
    {
        degree.push_back(_degree);
    }
}

PolynomialFunction::~PolynomialFunction()
{}

double PolynomialFunction::operator()(const vec& input)
{
    float result = 1.0;
    unsigned int i, j;
    for (i = 0; i < dimension.size(); i++)
    {
        for (j = 0; j < degree[i]; j++)
        {
            result *= input[dimension[i]];
        }
    }
    return result;
}

void PolynomialFunction::WriteOnStream(std::ostream &out)
{
    out << "Polynomial " << dimension.size() << std::endl;
    for (unsigned int i = 0; i < dimension.size(); i++)
    {
        out << dimension[i] << " " << degree[i] << std::endl;
    }
}

void PolynomialFunction::ReadFromStream(std::istream &in)
{
    unsigned int size = 0;
    in >> size;
    dimension.clear();
    degree.clear();
    unsigned int dim, deg;
    for (unsigned int i = 0; i < size; i++)
    {
        in >> dim >> deg;
        dimension.push_back(dim);
        degree.push_back(deg);
    }
}

}//end namespace
