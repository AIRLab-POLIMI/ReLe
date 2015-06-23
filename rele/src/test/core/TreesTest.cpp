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

#include "regressors/KDTree.h"
#include "regressors/ExtraTree.h"
#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"

#include <iostream>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    arma::vec defaultValue = {0};
    BasisFunctions basis = IdentityBasis::generate(1);
    DenseFeatures phi(basis);

    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> tree(phi, defaultNode);

    BatchDataPlain<arma::vec, arma::vec> dataset;
    tree.train(dataset);


    //Test the tree
    arma::vec input1 = {0};
    cout << "tree({0}) =" << endl;
    cout << tree.evaluate(input1) << endl;
}
