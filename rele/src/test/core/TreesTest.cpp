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
#include "regressors/ExtraTreeEnsemble.h"
#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"

#include <iostream>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    arma::vec defaultValue = {0};
    BasisFunctions basis = IdentityBasis::generate(2);
    DenseFeatures phi(basis);

    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> tree(phi, defaultNode);


    cout << "tree outputSize: " << tree.getOutputSize() << endl;

    BatchDataPlain<arma::vec, arma::vec> dataset;
    tree.train(dataset);


    //Test the tree
    arma::vec input1 = {0, 0};
    cout << "tree({0}) =" << endl;
    cout << tree(input1) << endl;


    //Train with atan2
    BatchDataPlain<arma::vec, arma::vec> datasetAtan2;
    for(int i = 0; i < 600; i++)
    {
        double step = 0.01;
        double angle = step*i;

        arma::vec input = {sin(angle), cos(angle)};
        arma::vec output = {atan2(sin(angle), cos(angle))};

        datasetAtan2.addSample(input, output);
    }

    tree.train(datasetAtan2);

    arma::vec test(2);
    test(0) = sin(M_PI/4);
    test(1) = cos(M_PI/4);
    cout << "tree =" << tree(test) << "gt = " << atan2(test(0), test(1)) << endl;

    test(0) = sin(M_PI/3);
    test(1) = cos(M_PI/3);
    cout << "tree =" << tree(test) << "gt = " << atan2(test(0), test(1)) <<  endl;


    double rAngle = RandomGenerator::sampleUniform(0, 2*M_PI);
    cout << "random Angle: " << rAngle << endl;
    test(0) = sin(rAngle);
    test(1) = cos(rAngle);
    cout << "tree =" << tree(test) << "gt = " << atan2(test(0), test(1)) <<  endl;

    cout << "### ExtraTreeTest ###" << endl;

    //ExtraTree test
    ExtraTreeEnsemble<arma::vec, arma::vec> extraTree(phi, defaultNode);

    extraTree.train(datasetAtan2);

    test(0) = sin(M_PI/4);
    test(1) = cos(M_PI/4);
    cout << "extraTree =" << extraTree(test) << "gt = " << atan2(test(0), test(1)) << endl;

    test(0) = sin(M_PI/3);
    test(1) = cos(M_PI/3);
    cout << "extraTree =" << extraTree(test) << "gt = " << atan2(test(0), test(1)) <<  endl;

    cout << "random Angle: " << rAngle << endl;
    test(0) = sin(rAngle);
    test(1) = cos(rAngle);
    cout << "extraTree =" << extraTree(test) << "gt = " << atan2(test(0), test(1)) <<  endl;
}
