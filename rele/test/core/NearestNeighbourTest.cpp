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

//#include "regressors/NearestNeighbourRegressor.h"
#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"

using namespace ReLe;

int main(int argc, char *argv[])
{
	/*unsigned int featureSize = 3;
	BasisFunctions basis = IdentityBasis::generate(featureSize);
	DenseFeatures phi(basis);
	NearestNeighbourRegressor regressor(phi, 2);

	arma::mat data1(featureSize, 10, arma::fill::randn);
	arma::mat data2(featureSize, 10, arma::fill::randn);
	data2 + 10;

	arma::mat data = arma::join_horiz(data1, data2);

	std::vector<arma::vec> vectorData;
	for(unsigned i = 0; i < data.n_cols; i++)
	{
		vectorData.push_back(data.col(i));
	}

	regressor.train(vectorData);*/

}
