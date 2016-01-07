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

#include "rele/approximators/regressors/others/NearestNeighbourRegressor.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"

using namespace ReLe;

int main(int argc, char *argv[])
{
    //Easy test
    unsigned int featureSize = 3;
    BasisFunctions basis = IdentityBasis::generate(featureSize);
    DenseFeatures phi(basis);
    NearestNeighbourRegressor regressor(phi, 2);

    arma::mat data_1(featureSize, 10, arma::fill::randn);
    arma::mat data_2(featureSize, 10, arma::fill::randn);
    data_2 += 10;

    arma::mat data = arma::join_horiz(data_1, data_2);

    std::vector<arma::vec> vectorData;
    for(unsigned i = 0; i < data.n_cols; i++)
    {
        vectorData.push_back(data.col(i));
    }

    regressor.train(vectorData);

    std::cout << "-- Test #1 --" << std::endl;
    std::cout << "data" << std::endl << data <<std::endl;
    std::cout << "computed centroids" << std::endl << regressor.getCentroids() << std::endl;
    std::cout << "computed clusters" << std::endl << arma::umat(regressor.getClusters()) << std::endl;
    std::cout << "wcss" << std::endl <<  regressor.getWCSS() << std::endl;


    //Test with possible local minima
    arma::mat data2 =
    {
        {-0.9994,    0.0113,   -0.0000,    0.0342},
        {-0.0358,   -0.2098,   -0.0000,   -0.9771}
    };

    std::vector<arma::vec> vectorData2;
    for(unsigned i = 0; i < data2.n_cols; i++)
    {
        vectorData2.push_back(data2.col(i));
    }

    BasisFunctions basis2 = IdentityBasis::generate(2);
    DenseFeatures phi2(basis2);
    NearestNeighbourRegressor regressor2(phi2, 2);

    regressor2.setIterations(5);
    regressor2.train(vectorData2);

    std::cout << "-- Test #2 --" << std::endl;
    std::cout << "data" << std::endl << data2 << std::endl;
    std::cout << "computed centroids" << std::endl << regressor2.getCentroids() << std::endl;
    std::cout << "computed clusters" << std::endl << arma::umat(regressor2.getClusters()) << std::endl;
    std::cout << "wcss" << std::endl <<  regressor2.getWCSS() << std::endl;



}
