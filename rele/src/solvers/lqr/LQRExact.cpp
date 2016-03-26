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

#include "rele/solvers/lqr/LQRExact.h"

namespace ReLe
{

LQRExact::LQRExact(double gamma, arma::mat A,
                   arma::mat B,
                   std::vector<arma::mat> Q,
                   std::vector<arma::mat> R,
                   arma::vec x0) : gamma(gamma), B(B), Q(Q), R(R), x0(x0)
{
	n_rewards = Q.size();
	n_dim = A.n_rows;
}

arma::mat LQRExact::computeP(const arma::mat& K, unsigned int r)
{

}

arma::mat LQRExact::riccatiRHS(const arma::vec& k, const arma::mat& P, unsigned int r)
{

}

arma::mat LQRExact::computeJ(const arma::mat& K, const arma::mat& Sigma)
{

}

arma::mat LQRExact::computeGradient(const arma::mat& K, const arma::mat& Sigma, unsigned int r)
{

}

arma::mat LQRExact::computeJacobian(const arma::mat& K, const arma::mat& Sigma)
{

}

arma::mat LQRExact::computeHesian(const arma::mat& K, const arma::mat& Sigma, unsigned int r)
{

}

arma::mat LQRExact::computeM(const arma::mat& K)
{

}

arma::mat LQRExact::compute_dM(const arma::mat& K, unsigned int i)
{

}

arma::mat LQRExact::computeHM(unsigned int i, unsigned int j)
{

}

arma::mat LQRExact::computeL(const arma::mat& K, unsigned int r)
{

}

arma::mat LQRExact::compute_dL(const arma::mat& K, unsigned int r, unsigned int i)
{

}

arma::mat LQRExact::computeHL(unsigned int r, unsigned int i, unsigned int j)
{

}

}
