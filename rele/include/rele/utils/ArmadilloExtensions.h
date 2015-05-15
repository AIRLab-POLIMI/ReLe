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

//=====================================================================
// additional function usefull for armadillo
// functions are taken from the MATLAB/OCTAVE versions
//---------------------------------------------------------------------
namespace ReLe
{

/**
 * @brief Null Space
 * @param A the input matrix
 * @param tol tolerance used in the rank test. If <0, a default tolerance is used
 * @return an orthonormal basis (Z) for the null space of A obtained from the singular value decomposition.
 * That is, A*Z has negligible elements, Z.n_cols is the nullity of A, and Z.t()*Z = I.
 */
arma::mat null(const arma::mat& A, double tol = -1);

/**
 * @brief Reduced row echelon form (Gauss-Jordan elimination)
 * @param X matrix to be reduced
 * @param A reduced row echelon form
 * @param tol tolerance used in the rank test. If <0, a default tolerance of (max(size(A))*eps *norm(A,inf)) tests for negligible column elements.
 * @return a vector idxs such that: 1) r = length(idxs) is this algorithm's idea of the rank of X. 2) A(:,idxs) is a basis for the range of X.
 * 3) A(1:r,idxs) is the r-by-r identity matrix.
 * In addition the reduced row echelon form of X using Gauss Jordan elimination with partial pivoting is returned through A.
 */
arma::uvec rref(const arma::mat& X, arma::mat& A, double tol = -1);

/**
 * @brief Wrap angle in radians to [−pi pi]
 * @param lambda the angle in radians
 * @return  wraps angle lambda, in radians, to the interval [−pi pi].
 */
double wrapTo2Pi(double lambda);
/**
 * @brief Wrap angle in radians to [−pi pi]
 * @param lambda the vector of angles to be wrapped
 * @return  wraps angles in lambda, in radians, to the interval [−pi pi].
 */
arma::vec wrapTo2Pi(const arma::vec& lambda);

/**
 * @brief Wrap angle in radians to [0 2*pi]
 * @param lambda the angle in radians
 * @return  wraps angle lambda, in radians, to the interval [0 2*pi].
 */
double wrapToPi(double lambda);
/**
 * @brief Wrap angle in radians to [0 2*pi]
 * @param lambda the vector of angles to be wrapped
 * @return  wraps angles in lambda, in radians, to the interval [0 2*pi].
 */
arma::vec wrapToPi(const arma::vec& lambda);

}

#endif //ARMADILLO_EXTENSIONS_H_
