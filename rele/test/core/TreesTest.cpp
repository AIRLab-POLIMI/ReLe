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

#include "rele/approximators/regressors/trees/KDTree.h"
#include "rele/approximators/regressors/trees/ExtraTree.h"
#include "rele/approximators/regressors/trees/ExtraTreeEnsemble.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"

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

    //Test the tree
    arma::vec input1 = {0, 0};
    cout << "tree({0}) =" << endl;
    cout << tree(input1) << endl;


    //Train with atan2
    arma::mat input(2, 1200);
    arma::mat output(1, 1200);

    for(int i = 0; i < 1200; i++)
    {
        double step = 0.005;
        double angle = step*i;

        input.col(i) = arma::vec({sin(angle), cos(angle)});
        output.col(i) = arma::vec({atan2(sin(angle), cos(angle))});
    }

    BatchDataSimple datasetAtan2(input, output);

    tree.trainFeatures(datasetAtan2);

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
    ExtraTreeEnsemble_<arma::vec, arma::vec> extraTree(phi, defaultNode);

    extraTree.trainFeatures(datasetAtan2);

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
