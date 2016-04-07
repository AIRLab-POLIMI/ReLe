/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/nn/Autoencoder.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    cout << "## Autoencoder Test ##" << endl;

    BasisFunctions basis = IdentityBasis::generate(4);
    DenseFeatures phi(basis);

    Autoencoder encoder(phi, 3);

    arma::mat numbers(3, 1000, arma::fill::randn);

    arma::mat features1 = numbers.row(0) + numbers.row(1);
    arma::mat features2 = numbers.row(2) % numbers.row(1);
    arma::mat features3 = numbers.row(2) - numbers.row(0);
    arma::mat features4 = numbers.row(0) + numbers.row(1) + numbers.row(2);

    arma::mat features(4, 1000);

    features.row(0) = features1;
    features.row(1) = features2;
    features.row(2) = features3;
    features.row(3) = features4;

    encoder.getHyperParameters().optimizator = new ScaledConjugateGradient<arma::vec>(5000);
    encoder.getHyperParameters().lambda = 0;


    std::cout << "J0 = " << encoder.computeJFeatures(features) << std::endl;
    encoder.trainFeatures(features);
    std::cout << "J  = " << encoder.computeJFeatures(features) << std::endl;

    arma::mat testNumbers(3, 5, arma::fill::randn);

    arma::mat testFeatures1 = testNumbers.row(0) + testNumbers.row(1);
    arma::mat testFeatures2 = testNumbers.row(2) % testNumbers.row(1);
    arma::mat testFeatures3 = testNumbers.row(2) - testNumbers.row(0);
    arma::mat testFeatures4 = testNumbers.row(0) + testNumbers.row(1) + testNumbers.row(2);

    arma::mat testFeatures(4, 5);

    testFeatures.row(0) = testFeatures1;
    testFeatures.row(1) = testFeatures2;
    testFeatures.row(2) = testFeatures3;
    testFeatures.row(3) = testFeatures4;

    std::cout << testFeatures << std::endl;

    for(unsigned int i = 0; i < testNumbers.n_cols; i++)
    {
        std::cout << "base           = " << testNumbers.col(i).t();
        std::cout << "input          = " << testFeatures.col(i).t();
        std::cout << "features       = " << encoder(testFeatures.col(i)).t();
        std::cout << "reconstructed  = " << encoder.FFNeuralNetwork::operator()(testFeatures.col(i)).t();
        std::cout << std::endl;
    }

}
