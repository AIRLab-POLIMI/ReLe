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

#include "LQRsolver.h"

using namespace arma;
using namespace std;

namespace ReLe
{

LQRsolver::LQRsolver(LQR& lqr, LinearApproximator& approximator) :
    lqr(lqr), pi(&approximator)
{
    rewardIndex = 0;
    gamma = lqr.getSettings().gamma;
}

void LQRsolver::solve()
{
    mat A = lqr.A;
    mat B = lqr.B;

    mat Q =lqr.Q[rewardIndex];
    mat R =lqr.R[rewardIndex];

    mat P(Q.n_rows, Q.n_cols, fill::eye);
    mat K;

    for(unsigned int j = 0; j < 100; j++)
    {
        K = -gamma*inv((R+gamma*(B.t()*P*B)))*B.t()*P*A;
        P = Q + gamma*A.t()*P*A + gamma*K.t()*B.t()*P*A
            + gamma*A.t()*P*B*K + gamma*K.t()*B.t()*P*B*K + K.t()*R*K;
    }

    K = -gamma*inv((R+gamma*(B.t()*P*B)))*B.t()*P*A;

    arma::vec w = vectorise(K);
    pi.setParameters(w);
}

Dataset<DenseAction, DenseState> LQRsolver::test()
{
    return Solver<DenseAction, DenseState>::test(lqr, pi);
}

Policy<DenseAction, DenseState>& LQRsolver::getPolicy()
{
    return pi;
}

}
