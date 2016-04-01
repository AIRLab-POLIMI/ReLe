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

#ifndef INCLUDE_RELE_SOLVERS_LQR_LQREXACT_H_
#define INCLUDE_RELE_SOLVERS_LQR_LQREXACT_H_

#include <armadillo>
#include "rele/environments/LQR.h"

namespace ReLe
{

/*!
 * This class is not strictly a solver, but it can be used to calculate the exact expected return,
 * the exact gradient, and the exact hessian of any LQR problem, and thus can be used by any optimization
 * algorithm to find the optimal value for this kind of problem, even in the multidimensional reward setting.
 */
class LQRExact
{
public:
    /*!
     * Constructor.
     * \param gamma the discout factor
     * \param A the state dynamics matrix
     * \param B the action dynamics matrix
     * \param Q a vector of weights matrixes for the state
     * \param R a vector of weights matrixes for the action
     * \param x0 the initial state for the LQR problem
     */
    LQRExact(double gamma, arma::mat A,
             arma::mat B,
             std::vector<arma::mat> Q,
             std::vector<arma::mat> R,
             arma::vec x0);

    /*!
     * Constructor.
     * \param lqr the LQR environment to be considered
     */
    LQRExact(LQR& lqr);


    /*!
     * Solves the Riccati equation for a given parameters vector.
     * \param k the weights vector
     * \param r the reward index
     * \return the solution to the Riccati equation
     */
    arma::mat solveRiccati(const arma::vec& k, unsigned int r = 0);

    /*!
     * Compute the right hand side of the Riccati equation for a given parameters vector.
     * \param k the weights vector
     * \param r the reward index
     * \return the value of the right hand side of the Riccati equation
     */
    arma::mat riccatiRHS(const arma::vec& k, const arma::mat& P, unsigned int r = 0);

    /*!
     * Compute the expected return under a normal policy.
     * \param k the weights vector
     * \param Sigma the covariance matrix
     * \return the expected return vector
     */
    arma::vec computeJ(const arma::vec& k, const arma::mat& Sigma);

    /*!
     * Compute the gradient of the expected return under a normal policy, w.r.t the parameters k.
     * \param k the weights vector
     * \param Sigma the covariance matrix
     * \param r the reward index
     * \return the gradient of the r component of the expected reward
     */
    arma::mat computeGradient(const arma::vec& k, const arma::mat& Sigma, unsigned int r = 0);

    /*!
     * Compute the jacobian of the expected return under a normal policy, w.r.t the parameters k.
     * \param k the weights vector
     * \param Sigma the covariance matrix
     * \return the jacobian of the expected reward
     */
    arma::mat computeJacobian(const arma::vec& k, const arma::mat& Sigma);

    /*!
     * Compute the hessian of the expected return under a normal policy, w.r.t the parameters k.
     * \param k the weights vector
     * \param Sigma the covariance matrix
     * \param r the reward index
     * \return the hessian of the expected return
     */
    arma::mat computeHesian(const arma::vec& k, const arma::mat& Sigma, unsigned int r = 0);


private:
    arma::mat computeP(const arma::mat& K, unsigned int r);

    arma::mat computeM(const arma::mat& K);
    arma::mat compute_dM(const arma::mat& K, unsigned int i);
    arma::mat computeHM(unsigned int i, unsigned int j);

    arma::mat computeL(const arma::mat& K, unsigned int r);
    arma::mat compute_dL(const arma::mat& K, unsigned int r, unsigned int i);
    arma::mat computeHL(unsigned int r, unsigned int i, unsigned int j);

    inline arma::vec to_vec(const arma::mat& m)
    {
        arma::vec v = reshape(m, n_dim*n_dim, 1);
        return v;
    }

    inline arma::mat to_mat(const arma::vec& v)
    {
        arma::mat M = reshape(v, n_dim, n_dim);
        return M;
    }

private:
    unsigned int n_rewards;
    unsigned int n_dim;
    double gamma;
    arma::mat A;
    arma::mat B;
    std::vector<arma::mat> Q;
    std::vector<arma::mat> R;
    arma::vec x0;
};

}

#endif /* INCLUDE_RELE_SOLVERS_LQR_LQREXACT_H_ */
