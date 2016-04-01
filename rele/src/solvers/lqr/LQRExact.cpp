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

#include <cassert>

using namespace arma;

namespace ReLe
{

LQRExact::LQRExact(double gamma, mat A,
                   mat B,
                   std::vector<mat> Q,
                   std::vector<mat> R,
                   vec x0) : gamma(gamma), A(A), B(B), Q(Q), R(R), x0(x0)
{
    n_rewards = Q.size();
    n_dim = A.n_rows;
}

LQRExact::LQRExact(LQR& lqr) :
    LQRExact(lqr.getSettings().gamma, lqr.A, lqr.B, lqr.Q, lqr.R, lqr.initialState)
{
}

mat LQRExact::solveRiccati(const vec& k, unsigned int r)
{
    mat K = -diagmat(k);
    return computeP(K, r);
}

mat LQRExact::riccatiRHS(const vec& k, const mat& P, unsigned int r)
{
    mat K = -diagmat(k);
    return Q[r] + gamma*(A.t()*P*A-K*B.t()*P*A-A.t()*P*B*K.t()+K*B.t()*P*B*K.t())+K*R[r]*K.t();
}

vec LQRExact::computeJ(const vec& k, const mat& Sigma)
{
    arma::vec J = zeros(n_rewards);

    mat K = -diagmat(k);

    for (unsigned int r = 0; r < n_rewards; r++)
    {
        mat P = computeP(K, r);
        J(r) = -as_scalar(x0.t()*P*x0 + trace(Sigma*(R[r] + gamma*B.t()*P*B))/(1.0-gamma));
    }

    return J;
}

mat LQRExact::computeGradient(const vec& k, const mat& Sigma, unsigned int r)
{
    assert(r < n_rewards);

    mat K = -diagmat(k);

    auto&& M = computeM(K);
    auto&& L = computeL(K, r);
    auto&& Minv = inv(M);

    arma::vec dJ = zeros(n_dim);
    for (unsigned int i = 0; i < n_dim; i++)
    {
        auto&& dMi = compute_dM(K, i);
        auto&& dLi = compute_dL(K, r, i);

        auto&&  vec_dPi = -Minv*dMi*Minv*to_vec(L)+solve(M, to_vec(dLi));

        auto&& dPi = to_mat(vec_dPi);

        dJ(i) = -as_scalar(x0.t()*dPi*x0+gamma*trace(Sigma*B.t()*dPi*B)/(1.0-gamma));
    }

    return -dJ;
}

mat LQRExact::computeJacobian(const vec& k, const mat& Sigma)
{
    arma::mat dJ = zeros(n_rewards, n_dim);

    for (unsigned int r=0; r < n_rewards; r++)
        dJ.row(r) = computeGradient(k, Sigma, r).t();

    return dJ;
}

mat LQRExact::computeHesian(const vec& k, const mat& Sigma, unsigned int r)
{
    assert(r < n_rewards);

    mat K = -diagmat(k);

    mat HJ = zeros(n_dim, n_dim);

    auto&& M = computeM(K);
    auto&& L = computeL(K, r);

    auto&& vecL = to_vec(L);

    mat Minv = inv(M);

    for (unsigned int i = 0; i < n_dim; i++)
    {
        auto&& dMi = compute_dM(K, i);
        auto&& dLi = compute_dL(K, r, i);
        auto&& vec_dLi = to_vec(dLi);

        for(unsigned int j = 0; j < n_dim; j++)
        {
            auto&& dMj = compute_dM(K, j);

            auto&& HMij = computeHM(i, j);
            auto&& HLij = computeHL(r, i, j);

            auto&& dLj = compute_dL(K, r, j);
            auto&& vec_dLj = to_vec(dLj);


            auto&& dMjinv = -Minv*dMj*Minv;

            vec vecHP = -dMjinv*dMi*Minv*vecL - Minv*dMi*dMjinv*vecL
                        -Minv*HMij*Minv*vecL  - Minv*dMi*Minv*vec_dLj
                        +dMjinv*vec_dLi       + Minv*to_vec(HLij);

            auto&& HP = to_mat(vecHP);

            HJ(i,j) = -as_scalar(x0.t()*HP*x0+gamma*trace(Sigma*B.t()*HP*B)/(1.0-gamma));
        }
    }

    return HJ;
}

mat LQRExact::computeP(const mat& K, unsigned int r)
{
    assert(r < n_rewards);

    auto&& L = computeL(K, r);
    auto&& M = computeM(K);

    vec vecP = solve(M, to_vec(L));

    mat P = to_mat(vecP);

    return P;
}

mat LQRExact::computeM(const mat& K)
{
    auto&& kb = K*B.t();
    unsigned int size = n_dim*n_dim;

    return eye(size, size) - gamma*(kron(A.t(), A.t()) - kron(A.t(), kb) - kron(kb, A.t()) + kron(kb, kb));
}

mat LQRExact::compute_dM(const mat& K, unsigned int i)
{
    assert(i < n_dim);

    arma::mat dKi = zeros(n_dim, n_dim);
    dKi(i, i) = 1;

    auto&& kb = K*B.t();
    auto&& dkb = dKi*B.t();

    return gamma*(kron(A.t(), dkb) + kron(dkb, A.t()) - kron(dkb, kb) - kron(kb, dkb));
}

mat LQRExact::computeHM(unsigned int i, unsigned int j)
{
    assert(i < n_dim);
    assert(j < n_dim);

    arma::mat dKi = zeros(n_dim, n_dim);
    dKi(i, i) = 1;

    arma::mat dKj = zeros(n_dim, n_dim);
    dKj(j, j) = 1;

    return -gamma*(kron(dKi*B.t(), dKj*B.t()) + kron(dKj*B.t(), dKi*B.t()));
}

mat LQRExact::computeL(const mat& K, unsigned int r)
{
    assert(r < n_rewards);

    return Q[r] + K*R[r]*K.t();
}

mat LQRExact::compute_dL(const mat& K, unsigned int r, unsigned int i)
{
    assert(r < n_rewards);
    assert(i < n_dim);

    arma::mat dKi = zeros(n_dim, n_dim);
    dKi(i, i) = 1;

    return dKi*R[r]*K.t() + K*R[r]*dKi.t();
}

mat LQRExact::computeHL(unsigned int r, unsigned int i, unsigned int j)
{
    assert(r < n_rewards);
    assert(i < n_dim);
    assert(j < n_dim);

    arma::mat dKi = zeros(n_dim, n_dim);
    dKi(i, i) = 1;

    arma::mat dKj = zeros(n_dim, n_dim);
    dKj(j, j) = 1;

    return dKi*R[r]*dKj.t() + dKj*R[r]*dKi.t();
}

}
