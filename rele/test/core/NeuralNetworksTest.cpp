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

#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"
#include "regressors/nn/FFNeuralNetwork.h"

#include "Utils.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    cout << "## Neural Network Test ##" << endl;

    //Train plane
    BasisFunctions basisPlane = IdentityBasis::generate(1);
    DenseFeatures phiPlane(basisPlane);

    FFNeuralNetwork planeNet(phiPlane, 10, 1);

    BatchDataRaw_<arma::vec, arma::vec> datasetPlane;

    //Config parameters
    for(int i = -100; i < 100; i++)
    {
        double f = i;
        arma::vec input = { f };
        arma::vec output = { f };
        datasetPlane.addSample(input, output);
    }

    planeNet.getHyperParameters().normalizationF = new MinMaxNormalization<>();
    //planeNet.getHyperParameters().normalizationO = new MinMaxNormalization<>();
    planeNet.train(datasetPlane);

    cout << "plane(100)  = " << planeNet({100.0}) <<  endl;
    cout << "plane(50)   = " << planeNet({50.0}) << endl;
    cout << "plane(30)   = " << planeNet({30.0}) <<  endl;
    cout << "plane(0)    = " << planeNet({0.0}) << endl;
    cout << "plane(-30)  = " << planeNet({-30.0}) <<  endl;
    cout << "plane(-50)  = " << planeNet({-50.0}) << endl;
    cout << "plane(-100) = " << planeNet({-100.0}) <<  endl;

    arma::vec input = {1.0};
    arma::vec numerical = NumericalGradient::compute(planeNet, planeNet.getParameters(), input);
    cout << "numerical: " << numerical.t();
    cout << "exact:     " << planeNet.diff(input).t();
    cout << "n/e:       " << numerical.t()/planeNet.diff(input).t();
    cout << "error norm: " << norm(numerical - planeNet.diff(input)) << endl;

    cout << "J = " << planeNet.computeJ(datasetPlane) << endl;


    // Generate adequate basis
    BasisFunctions basis = IdentityBasis::generate(2);
    DenseFeatures phi(basis);


    //Train atan2
    cout << "atan2 test" << endl;
    FFNeuralNetwork atan2Net(phi, 20, 1);
    BatchDataRaw_<arma::vec, arma::vec> dataset;

    //Config parameters
    atan2Net.getHyperParameters().optimizator = new GradientDescend<arma::vec>(10000, 0.2);



    for(int i = 0; i < 300; i++)
    {
        double step = 0.01;
        double angle = step*i;

        arma::vec input = {sin(angle), cos(angle)};
        arma::vec output = {atan2(sin(angle), cos(angle))};

        dataset.addSample(input, output);

    }

    atan2Net.train(dataset);

    arma::vec test(2);
    test(0) = sin(M_PI/4);
    test(1) = cos(M_PI/4);
    cout << "net =" << atan2Net(test) << "gt = " << atan2(test(0), test(1)) << endl;

    test(0) = sin(M_PI/3);
    test(1) = cos(M_PI/3);
    cout << "net =" << atan2Net(test) << "gt = " << atan2(test(0), test(1)) <<  endl;

    cout << "J = " << atan2Net.computeJ(dataset) << endl;


    //Train xor
    FFNeuralNetwork xorNet(phi, 3, 1);
    BatchDataRaw_<arma::vec, arma::vec> datasetXor;

    //Config parameters
    xorNet.getHyperParameters().optimizator = new GradientDescend<arma::vec>(10000, 0.2);

    arma::vec i0 = {0.0, 0.0};
    arma::vec i1 = {1.0, 0.0};
    arma::vec i2 = {0.0, 1.0};
    arma::vec i3 = {1.0, 1.0};

    arma::vec o0 = {0.0};
    arma::vec o1 = {1.0};

    datasetXor.addSample(i0, o0);
    datasetXor.addSample(i1, o1);
    datasetXor.addSample(i2, o1);
    datasetXor.addSample(i3, o0);

    xorNet.train(datasetXor);

    cout << "xor(0, 0) =" << xorNet(i0) << "gt = " << "0" << endl;
    cout << "xor(1, 0) =" << xorNet(i1) << "gt = " << "1" <<  endl;
    cout << "xor(0, 1) =" << xorNet(i2) << "gt = " << "1" << endl;
    cout << "xor(1, 1) =" << xorNet(i3) << "gt = " << "0" <<  endl;

    cout << "J = " << xorNet.computeJ(datasetXor) << endl;

}

