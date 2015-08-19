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
#include "regressors/FFNeuralNetwork.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    cout << "## Neural Network Test ##" << endl;

    arma::vec input = {1.0, 1.0};

    BasisFunctions basis = IdentityBasis::generate(input.size());
    DenseFeatures phi(basis);

    FFNeuralNetwork net(phi, 3, 1);
    arma::vec p = net.getParameters();

    arma::vec out1 = net(input);
    cout << "f(1.0, 1.0) =" << out1 << endl;

    p.fill(1.0);
    net.setParameters(p);
    arma::vec out2 = net(input);
    cout << "f(1.0, 1.0) =" << out2 << endl;

    arma::vec diff = net.diff(input);
    cout << "f'(1.0, 1.0) =" << diff.t() << endl;


    //Train atan2
    FFNeuralNetwork atan2Net(phi, 5, 1);
    BatchDataPlain<arma::vec, arma::vec> dataset;

    //Config parameters
    atan2Net.getHyperParameters().alg = FFNeuralNetwork::Adadelta;
    atan2Net.getHyperParameters().epsilon = 0.01;
    atan2Net.getHyperParameters().rho = 0.1;
    atan2Net.getHyperParameters().maxIterations = 2500;
    atan2Net.getHyperParameters().lambda = 0;
    atan2Net.getHyperParameters().minibatchSize = dataset.size();

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

    cout << "J = " << atan2Net.computeJ(dataset, 0) << endl;


    //Train xor
    FFNeuralNetwork xorNet(phi, 3, 1);
    BatchDataPlain<arma::vec, arma::vec> datasetXor;

    //Config parameters
    xorNet.getHyperParameters().alg = FFNeuralNetwork::GradientDescend;
    xorNet.getHyperParameters().alpha = 0.6;
    xorNet.getHyperParameters().maxIterations = 14000;
    xorNet.getHyperParameters().lambda = 0;

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

    arma::vec p_0(xorNet.getParametersSize(), arma::fill::ones);
    xorNet.setParameters(p_0);
    xorNet.train(datasetXor);

    cout << "xor(0, 0) =" << xorNet(i0) << "gt = " << "0" << endl;
    cout << "xor(1, 0) =" << xorNet(i1) << "gt = " << "1" <<  endl;
    cout << "xor(0, 1) =" << xorNet(i2) << "gt = " << "1" << endl;
    cout << "xor(1, 1) =" << xorNet(i3) << "gt = " << "0" <<  endl;

    cout << "J = " << xorNet.computeJ(datasetXor, 0) << endl;

}

