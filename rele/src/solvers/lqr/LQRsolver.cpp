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

#include "rele/solvers/lqr/LQRsolver.h"
#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe
{

LQRsolver::LQRsolver(LQR& lqr, Features& phi, Type type) :
    lqr(lqr), pi(phi), weightsRew(lqr.getSettings().rewardDimensionality, arma::fill::zeros),
    solution_type(type)
{
    weightsRew(0) = 1;
    gamma = lqr.getSettings().gamma;
}

void LQRsolver::solve()
{

    arma::mat K = computeOptSolution();
    arma::vec w = arma::vectorise(K.t());
    pi.setParameters(w);
}

arma::mat LQRsolver::computeOptSolution()
{
    mat A = lqr.A;
    mat B = lqr.B;

    mat K;
    mat Q(lqr.Q[0].n_rows, lqr.Q[0].n_cols, arma::fill::zeros);
    mat R(lqr.R[0].n_rows, lqr.R[0].n_cols, arma::fill::zeros);

    if (solution_type == MOO)
    {
        int dim = lqr.Q.size();
        assert(weightsRew.n_elem == dim);

        for (int i = 0; i < dim; ++i)
        {
            Q += weightsRew[i] * lqr.Q[i];
            R += weightsRew[i] * lqr.R[i];
        }

    }
    else
    {
        int dimS = lqr.A.n_cols;
        int dimA = lqr.B.n_cols;
        int dim  = lqr.Q.size();
        assert(weightsRew.n_elem == dim*(dimS*dimS+dimA*dimA));

        int cont = 0;
        for (int i = 0; i < dim; ++i)
        {
            for (int c = 0; c < dimS; ++c)
            {
                for (int r = 0; r < dimS; ++r)
                {
                    Q(r,c) += weightsRew[cont++];
                }
            }

            for (int c = 0; c < dimA; ++c)
            {
                for (int r = 0; r < dimA; ++r)
                {
                    R(r,c) += weightsRew[cont++];
                }
            }
        }

//        std::cout << Q << std::endl;
//        std::cout << R << std::endl;
    }

    mat P(Q.n_rows, Q.n_cols, fill::eye);
    for(unsigned int j = 0; j < 100; j++)
    {
        K = -gamma*inv((R+gamma*(B.t()*P*B)))*B.t()*P*A;
        P = Q + gamma*A.t()*P*A + gamma*K.t()*B.t()*P*A
            + gamma*A.t()*P*B*K + gamma*K.t()*B.t()*P*B*K + K.t()*R*K;
    }

    K = -gamma*inv((R+gamma*(B.t()*P*B)))*B.t()*P*A;

    return K;
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
