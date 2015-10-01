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

#ifndef CONDITIONBASISFUNCTION_H_
#define CONDITIONBASISFUNCTION_H_

#include "BasisFunctions.h"

namespace ReLe
{

class AndConditionBasisFunction: public BasisFunction
{
public:
    AndConditionBasisFunction(BasisFunction* bfs, const std::vector<unsigned int>& idxs,
                              const std::vector<double>& condition_vals);

    AndConditionBasisFunction(BasisFunction* bfs, std::initializer_list<unsigned int>  idxs,
                              std::initializer_list<double> condition_vals);
    AndConditionBasisFunction(BasisFunction* bfs, unsigned int idx, double condition_vals);

    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;

    static BasisFunctions generate(BasisFunctions& basis, unsigned int index, unsigned int value);
    static BasisFunctions generate(BasisFunctions& basis, std::vector<unsigned int> indexes,
                                   std::vector<unsigned int> valuesVector);

private:
    static void generateRecursive(BasisFunctions& basis,
                                  const std::vector<unsigned int>& indexes,
                                  const std::vector<unsigned int>& valuesVector,
                                  std::vector<double>& currentValues, unsigned int currentindex,
                                  BasisFunctions& newBasis);

private:
    BasisFunction* basis;
    std::vector<unsigned int> idxs;
    std::vector<double> values;
};

}//end namespace


#endif // CONDITIONBASISFUNCTION_H_
