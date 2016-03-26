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

#ifndef INCLUDE_RELE_SOLVERS_LQRSOLVER_CPP_
#define INCLUDE_RELE_SOLVERS_LQRSOLVER_CPP_

#include "rele/core/Solver.h"
#include "rele/environments/LQR.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"

namespace ReLe
{

class LQRsolver: public Solver<DenseAction, DenseState>
{
public:
    enum Type {MOO, CLASSIC};

    LQRsolver(LQR& lqr, Features& phi, Type type = Type::MOO);
    virtual void solve() override;
    virtual Dataset<DenseAction, DenseState> test() override;
    virtual Policy<DenseAction, DenseState>& getPolicy() override;

    arma::mat computeOptSolution();
    inline void setRewardIndex(unsigned int rewardIndex)
    {
        weightsRew.zeros();
        weightsRew[rewardIndex] = 1;
    }

    inline void setRewardWeights(arma::vec& weights)
    {
        weightsRew = weights;
    }

protected:
    LQR& lqr;
    DetLinearPolicy<DenseState> pi;
    double gamma;
    arma::vec weightsRew;
    Type solution_type;
};

}

#endif /* INCLUDE_RELE_SOLVERS_LQRSOLVER_CPP_ */
