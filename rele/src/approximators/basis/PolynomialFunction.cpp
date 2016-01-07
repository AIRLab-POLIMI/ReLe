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

#include "rele/approximators/basis/PolynomialFunction.h"

#include <cassert>

using namespace arma;

namespace ReLe
{

PolynomialFunction::PolynomialFunction()
{
}

PolynomialFunction::PolynomialFunction(std::vector<unsigned int> dimension, std::vector<unsigned int> degree)
    : dimension(dimension), degree(degree)
{
}

PolynomialFunction::PolynomialFunction(unsigned int _dimension, unsigned int _degree)
    : dimension(1, _dimension)
{
    assert(_degree > 0);
    degree.push_back(_degree);
}

PolynomialFunction::~PolynomialFunction()
{
}

double PolynomialFunction::operator()(const vec& input)
{
    double result = 1.0;
    unsigned int i, j;
    for (i = 0; i < dimension.size(); i++)
    {
        for (j = 0; j < degree[i]; j++)
        {
            result *= input(dimension[i]);
        }
    }
    return result;
}

void PolynomialFunction::writeOnStream(std::ostream &out)
{
    out << "Polynomial " << dimension.size() << std::endl;
    for (unsigned int i = 0; i < dimension.size(); i++)
    {
        out << dimension[i] << " " << degree[i] << std::endl;
    }
}

void PolynomialFunction::readFromStream(std::istream &in)
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


BasisFunctions PolynomialFunction::generate(unsigned int degree, unsigned int input_size)
{
    BasisFunctions basis;

    std::vector<unsigned int> dim;
    for (unsigned int i = 0; i < input_size; i++)
    {
        dim.push_back(i);
    }
    for (unsigned int d = 0; d <= degree; d++)
    {
        std::vector<unsigned int> deg(input_size);
        deg[0] = d;
        generatePolynomialsPermutations(deg, dim, basis);
        generatePolynomials(deg, dim, 1, basis);
    }

    return basis;
}

void PolynomialFunction::generatePolynomialsPermutations(std::vector<unsigned int> deg,
        std::vector<unsigned int>& dim, BasisFunctions& basis)
{
    std::sort(deg.begin(), deg.end());
    do
    {
        BasisFunction* pBF = new PolynomialFunction(dim, deg);
        basis.push_back(pBF);
    }
    while (next_permutation(deg.begin(), deg.end()));
}

void PolynomialFunction::generatePolynomials(std::vector<unsigned int> deg,
        std::vector<unsigned int>& dim,
        unsigned int place, BasisFunctions& basis)
{
    if (deg.size() > 1)
    {
        if (deg[0] > deg[1] && deg[place] < deg[place - 1] && deg[0] - deg[place] > 1)
        {
            std::vector<unsigned int> degree = deg;
            degree[0]--;
            degree[place]++;
            generatePolynomialsPermutations(degree, dim, basis);
            generatePolynomials(degree, dim, place, basis);
            if (place < deg.size() - 1)
            {
                generatePolynomials(degree, dim, place + 1, basis);
            }
        }
    }
}

}//end namespace
