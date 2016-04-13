/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/statistics/Wishart.h"
#include "rele/utils/ArmadilloPDFs.h"

namespace ReLe
{

Wishart::Wishart(unsigned int p) :
	Distribution(p*p), nu(p), V(p, p, arma::fill::eye)
{

}

Wishart::Wishart(unsigned int p, unsigned int nu) :
			Distribution(p*p), nu(nu), V(p, p, arma::fill::eye)
{

}

Wishart::Wishart(unsigned int nu,
                 const arma::mat& V) :
			Distribution(V.n_rows*V.n_cols), nu(nu), V(V)
{

}

Wishart::~Wishart()
{

}

arma::vec Wishart::operator() () const
{
	unsigned int p = V.n_rows;
	arma::mat X = mvnrand(nu, arma::zeros(p), V);
	arma::mat Lambda = X*X.t();

	return arma::vectorise(Lambda);
}

double Wishart::operator() (const arma::vec& point) const
{
	unsigned int p = V.n_rows;
	arma::mat X = arma::reshape(point, p, p);

	double Z = std::pow(2, 0.5*nu*p)*std::pow(arma::det(V), 0.5*nu)*tgamma_p(p, 0.5*nu);

	return std::pow(arma::det(X), 0.5*(nu-p-1))*std::exp(-0.5*arma::trace(V.i()*X))/Z;

}

void Wishart::wmle(const arma::vec& weights, const arma::mat& samples)
{
    unsigned int p = V.n_rows;
    arma::vec vecV = samples*weights/arma::sum(weights);
    V = arma::reshape(vecV, p, p);
}

double Wishart::tgamma_p(unsigned int p, double value) const
{

	double v = std::pow(M_PI, 0.25*p*(p-1));
	for(unsigned int j = 0; j < p; j++)
	{
		v *= std::tgamma(value-0.5*j);
	}

	return v;
}

}
