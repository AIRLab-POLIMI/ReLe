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

/*!
 * Additional functions for useful
 * operations on armadillo variables.
 */
namespace ReLe
{

/*!
 * Null Space.
 * \param A the input matrix
 * \param tol tolerance used in the rank test. If tol < 0, a default tolerance is used
 * \return an orthonormal basis (Z) for the null space of A obtained from Singular Value Decomposition (SVD).
 * That is, A * Z has negligible elements, Z.n_cols is the nullity of A and Z.t() * Z = I.
 */
arma::mat null(const arma::mat& A, double tol = -1);

/*!
 * Reduced row echelon form (Gauss-Jordan elimination).
 * \param X matrix to be reduced
 * \param A reduced row echelon form
 * \param tol tolerance used in the rank test. If tol < 0, a default tolerance of (max(size(A)) * eps * norm(A, inf)) tests for negligible column elements.
 * \return a vector idxs such that:
 * 1. r = length(idxs) is this algorithm's idea of the rank of X.
 * 2. A(:, idxs) is a basis for the range of X.
 * 3. A(1:r, idxs) is the r-by-r identity matrix.
 * In addition the reduced row echelon form of X using Gauss-Jordan elimination with partial pivoting is returned through A.
 */
arma::uvec rref(const arma::mat& X, arma::mat& A, double tol = -1);

/*!
 * Wrap angle in radians to [−pi, pi].
 * \param lambda the angle in radians
 * \return wraps angle lambda (in radians) to the interval [−pi, pi]
 */
double wrapTo2Pi(double lambda);

/*!
 * Wrap angle in radians to [−pi, pi].
 * \param lambda the vector of angles to be wrapped
 * \return wraps angles in lambda (in radians) to the interval [−pi, pi]
 */
arma::vec wrapTo2Pi(const arma::vec& lambda);

/*!
 * Wrap angle in radians to [0, 2 * pi].
 * \param lambda the angle in radians
 * \return wraps angle lambda (in radians) to the interval [0, 2 * pi]
 */
double wrapToPi(double lambda);

/*!
 * Wrap angle in radians to [0, 2 * pi]
 * \param lambda the vector of angles to be wrapped
 * \return wraps angles in lambda (in radians) to the interval [0, 2 * pi]
 */
arma::vec wrapToPi(const arma::vec& lambda);

/*!
 * Rectangular grid in 2-D space. meshgrid(xgv, ygv, X, Y) replicates the grid vectors xgv and ygv to produce a full grid stored in X and Y.
 * This grid is represented by the output coordinate arrays X and Y.
 * The output coordinate arrays X and Y contain copies of the grid vectors xgv and ygv respectively.
 * The sizes of the output arrays are determined by the length of the grid vectors.
 * For grid vectors xgv and ygv of length M and N respectively, X and Y will have N rows and M columns.
 * \param x Grid vector specifying a series of grid point coordinates in the x direction
 * \param y Grid vector specifying a series of grid point coordinates in the y direction
 * \param xx Output matrix that specifies the full grid components in the x direction
 * \param yy Output matrix that specifies the full grid components in the y direction
 */
void meshgrid(const arma::vec& x, const arma::vec& y, arma::mat& xx, arma::mat& yy);

/*!
 * Generate a block diagonal matrix from the input arguments.
 * Note that the provided order is used to define the diagonal.
 * \param diag_blocks a set of matrices
 * \return a sparse block diagonal matrix
 */
arma::sp_mat blockdiagonal(const std::vector<arma::mat>& diag_blocks);

/*!
 * Generate a block diagonal matrix from the input arguments.
 * This implementation is more efficient when the size of the resulting matrix is known by the caller.
 * This function avoid the computation of such information from the given matrix vector.
 * \param diag_blocks
 * \param rows number of rows of the resulting matrix
 * \param cols number of cols of the resulting matrix
 * \return a sparse block diagonal matrix
 */
arma::sp_mat blockdiagonal(const std::vector<arma::mat>& diag_blocks, int rows, int cols);

/*!
 * range(X) returns the difference between the maximum and the minimum of a sample.
 * For vectors, range(x) is the range of the elements.
 * For matrices, range(X) is a row vector containing the range of each column of X.
 * \param X the vector of matrix to be analyzed
 * \param dim the range is computed along dimension dim of X
 * \return the difference between the maximum and the minimum
 */
arma::vec range(arma::mat& X, unsigned int dim = 0);

/*!
 * Generate a lower triangular matrix from the input vector.
 * This function avoids the computation of matrix size from the given matrix vector.
 * \param vector vector of elements of the triangular matrix, must be of size (dim^2 - dim) / 2
 * \param triangular resulting triangular matrix, must be of size (dim, dim)
 */
void vecToTriangular(const arma::vec& vector, arma::mat& triangular);

/*!
 * Generate a vector from the input lower triangular matrix.
 * This function avoid the computation of matrix size from the given vector.
 * \param triangular resulting triangular matrix, must be of size (dim, dim)
 * \param vector vector of nonzero elements of the triangular matrix, must be of size (dim^2 - dim) / 2
 */
void triangularToVec(const arma::mat& triangular, arma::vec& vector);

/*!
 * Perform Cholesky Decomposition using nearest SPD.
 * Nearest SPD finds the nearest (in Frobenius norm) Symmetric Positive Definite
 * matrix to A. It can be calculated with (B + H) / 2 where H is the symmetric
 * polar factor of B = (A + A') / 2.
 *
 * References
 * ==========
 * [Higham J., Nicholas. Computing a Nearest Symmetric Positive Semidefinite Matrix](http://www.sciencedirect.com/science/article/pii/0024379588902236)
 *
 * \param M the matrix to be decomposed
 * \return the decomposed matrix
 */
arma::mat safeChol(arma::mat& M);

//void meshgrid(const arma::vec& x, const arma::vec& y, const arma::vec& z, arma::mat& xx, arma::mat& yy, arma::mat& zz);

}

#endif //ARMADILLO_EXTENSIONS_H_
