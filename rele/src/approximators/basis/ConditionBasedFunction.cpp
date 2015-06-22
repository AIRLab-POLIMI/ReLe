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

#include "basis/ConditionBasedFunction.h"

#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe
{

AndConditionBasisFunction::AndConditionBasisFunction(
    BasisFunction *bfs,
    const vector<unsigned int>& idxs,
    const vector<double>& condition_vals)
    : basis(bfs), idxs(idxs), values(condition_vals)
{
}

AndConditionBasisFunction::AndConditionBasisFunction(
    BasisFunction *bfs,
    initializer_list<unsigned int> idxs,
    initializer_list<double> condition_vals)
    : basis(bfs), idxs(idxs), values(condition_vals)
{
}

AndConditionBasisFunction::AndConditionBasisFunction(
    BasisFunction *bfs, unsigned int idx, double condition_vals)
    : basis(bfs)
{
    idxs.push_back(idx);
    values.push_back(condition_vals);
}

double AndConditionBasisFunction::operator()(const arma::vec &input)
{
    unsigned int i, nbcond = idxs.size();
    for (i = 0; i < nbcond; ++i)
    {
        if (input[idxs[i]] != values[i])
            return 0.0;
    }
    return (*basis)(input);
}

void AndConditionBasisFunction::writeOnStream(ostream &out)
{
    out << "AndConditionBasisFunction";
    unsigned int i, nbcond = idxs.size();
    out << nbcond << endl;
    for (i = 0; i < nbcond; ++i)
    {
        out << idxs[i] << "\t";
    }
    out << endl;
    for (i = 0; i < nbcond; ++i)
    {
        out << values[i] << "\t";
    }
    out << endl;
    out << *basis;
}

void AndConditionBasisFunction::readFromStream(istream &in)
{
    unsigned int i, nbcond, val;
    in >> nbcond;
    for (i = 0; i < nbcond; ++i)
    {
        in >> val;
        idxs.push_back(val);
    }
    double dval;
    for (i = 0; i < nbcond; ++i)
    {
        in >> dval;
        values.push_back(val);
    }
    //TOFIX manca un factory per le basis (come faccio a leggere la basis function?)
}

BasisFunctions AndConditionBasisFunction::generate(BasisFunctions& basis, unsigned int index,
        unsigned int values)
{
    BasisFunctions newBasis;

    for (int i = 0; i < values; ++i)
    {
        for(int j = 0; j < basis.size(); j++)
        {
            newBasis.push_back(new AndConditionBasisFunction(basis[j], index, i));
        }
    }

    return newBasis;
}

BasisFunctions AndConditionBasisFunction::generate(BasisFunctions& basis, vector<unsigned int> indexes,
        vector<unsigned int> valuesVector)
{
    assert(valuesVector.size() == indexes.size());

    BasisFunctions newBasis;
    vector<double> currentValues(valuesVector.size());

    generateRecursive(basis, indexes, valuesVector, currentValues, 0, newBasis);

    return newBasis;
}

void AndConditionBasisFunction::generateRecursive(BasisFunctions& basis,
        const std::vector<unsigned int>& indexes, const std::vector<unsigned int>& valuesVector,
        std::vector<double>& currentValues, unsigned int currentIndex, BasisFunctions& newBasis)
{
    if(currentIndex == valuesVector.size())
    {
        for(unsigned int i = 0; i < basis.size(); i++)
        {
            newBasis.push_back(new AndConditionBasisFunction(basis[i], indexes, currentValues));
        }
    }
    else
    {
        unsigned int values = valuesVector[currentIndex];

        for(unsigned int value = 0; value < values; value++)
        {
            currentValues[currentIndex] = value;
            generateRecursive(basis, indexes, valuesVector, currentValues, currentIndex + 1, newBasis);
        }

    }

}


}//end namespace
