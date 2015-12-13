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

#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"
#include "regressors/nn/Autoencoder.h"

#include "Utils.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    cout << "## Autoencoder Test ##" << endl;

    BasisFunctions basis = IdentityBasis::generate(2);
    DenseFeatures phi(basis);

    Autoencoder encoder(phi, 1);

    arma::rowvec angles = arma::linspace<arma::rowvec>(0, 2*M_PI, 1000);

    arma::mat features = arma::join_vert(arma::sin(angles), arma::cos(angles));
    std::cout << "J0 = " << encoder.computeJFeatures(features) << std::endl;

    encoder.getHyperParameters().alg = FFNeuralNetwork::GradientDescend;
    encoder.getHyperParameters().alpha = 0.2;
    encoder.getHyperParameters().maxIterations = 10000;
    encoder.getHyperParameters().lambda = 0;

    encoder.trainFeatures(features);


    std::cout << "J = " << encoder.computeJFeatures(features) << std::endl;

    arma::rowvec testAngles(5, arma::fill::randn);
    arma::mat inputs = arma::join_vert(arma::sin(angles), arma::cos(angles));

    for(unsigned int i = 0; i < testAngles.n_elem; i++)
    {
    	std::cout << "angle          = " << testAngles(i) << std::endl;
    	std::cout << "input          = " << inputs.col(i).t();
    	std::cout << "features       = " << encoder(inputs.col(i)).t();
    	std::cout << "reconstructed  = " << encoder.FFNeuralNetwork::operator()(inputs.col(i)).t();
    }

}
