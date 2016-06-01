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

#include "rele/approximators/BasisFunctions.h"

//TODO [IMPORTANT][INTERFACE] Rename and refactor thiss class: ActionBasisFunctions, with always indx == input.tail()

namespace ReLe
{

/*!
 * This class implements functions to attach a target feature(s)
 * to a group of given basis functions. This is done by repeating
 * the group of basis functions a number of time equal to the number
 * of possible values of the target feature(s) and multiplying
 * all elements of each group by the respective value of the target
 * feature(s) and setting the others to zero.
 * In particular, this can be useful with finite action spaces
 * using the possible values of the action as the target feature.
 */
class AndConditionBasisFunction: public BasisFunction
{
public:
    /*!
     * Constructor.
     * \param bfs the basis functions to which the target features are attached
     * \param idxs vector of indexes of the target features
     * \param condition_vals vector of the number of possible values of the target features
     */
    AndConditionBasisFunction(BasisFunction* bfs, const std::vector<unsigned int>& idxs,
                              const std::vector<double>& condition_vals);

    /*!
     * Constructor.
     * \param bfs the basis functions to which the target features are attached
     * \param idxs initializer list of indexes of the target features
     * \param condition_vals initializer list of the number of possible values of the target features
     */
    AndConditionBasisFunction(BasisFunction* bfs, std::initializer_list<unsigned int>  idxs,
                              std::initializer_list<double> condition_vals);

    /*!
     * Constructor.
     * \param bfs the basis functions to which the target feature is attached
     * \param idx the index of the target feature
     * \param condition_vals the number of possible values of the target feature
     */
    AndConditionBasisFunction(BasisFunction* bfs, unsigned int idx, double condition_vals);

    double operator()(const arma::vec& input) override;
    void writeOnStream(std::ostream& out) override;
    void readFromStream(std::istream& in) override;

    /*!
     * Generate the basis functions.
     * \param basis the basis functions to which the target feature is attached
     * \param index the index of the target feature
     * \param value the number of possible values of the target feature
     * \return the basis functions
     */
    static BasisFunctions generate(BasisFunctions& basis, unsigned int index, unsigned int value);

    /*!
     * Generate the basis functions.
     * \param basis the basis functions to which the target features are attached
     * \param indexes vector of indexes of the target features
     * \param valuesVector vector of the number of possible values of the target features
     * \return the basis functions
     */
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
