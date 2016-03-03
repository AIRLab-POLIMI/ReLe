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

#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/GaussianMixtureModels.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"

using namespace ReLe;
using namespace std;


int main(int argc, char *argv[])
{
    //Test EM
    unsigned int size = 2;
    unsigned int nSamples = 10000;
    std::vector<arma::vec> samples(nSamples);

    //generate data from gaussians
    arma::vec componentProbabilities = { 0.2, 0.3, 0.4, 0.1 };
    arma::mat componentMeans =        {{ 1.0, 1.0, 5.0, 5.0 },
        { 1.0, 5.0, 1.0, 5.0 }
    };

    for (unsigned int i = 0; i < nSamples; i++)
    {
        unsigned int k = RandomGenerator::sampleDiscrete(
                             componentProbabilities.begin(),
                             componentProbabilities.end());
        samples[i] = mvnrandFast(componentMeans.col(k),
                                 0.1 * arma::eye(size, size));
    }

    //launch EM
    BasisFunctions basis = IdentityBasis::generate(size);
    DenseFeatures phi(basis);
    GaussianMixtureRegressor regressor(phi, componentProbabilities.n_elem);
    regressor.train(samples);

    cout << regressor.getParameters().t();
}
