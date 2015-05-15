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

#ifndef ARMADILLO_EXTENSIONS_H_
#define ARMADILLO_EXTENSIONS_H_

#include <armadillo>

namespace ReLe
{

arma::mat null(const arma::mat& A, double tol = -1);

arma::uvec rref(const arma::mat& X, arma::mat& A, double tol = -1);

double wrapTo2Pi(double lambda);
arma::vec wrapTo2Pi(const arma::vec& lambda);

double wrapToPi(double lambda);
arma::vec wrapToPi(const arma::vec& lambda);

}

#endif //ARMADILLO_EXTENSIONS_H_
