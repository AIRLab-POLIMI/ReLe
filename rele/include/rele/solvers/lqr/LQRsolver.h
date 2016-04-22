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

/*!
 * This class implements the Linear-Quadratic Regulator (LQR) solver.
 */
class LQRsolver: public Solver<DenseAction, DenseState>
{
public:
    enum Type {MOO, CLASSIC};

    /*!
     * Constructor.
     * \param lqr the Linear-Quadratic Regulator
     * \param phi features
     * \param type the type of Linear-Quadratic Regulator
     */
    LQRsolver(LQR& lqr, Features& phi, Type type = Type::MOO);

    virtual void solve() override;
    virtual Dataset<DenseAction, DenseState> test() override;
    virtual Policy<DenseAction, DenseState>& getPolicy() override;

    /*!
     * Compute the optimal solution of the problem.
     * \return the matrix with the optimal solution
     */
    arma::mat computeOptSolution();

    /*!
     * Setter.
     * \param rewardIndex the index of the reward to set
     */
    inline void setRewardIndex(unsigned int rewardIndex)
    {
        weightsRew.zeros();
        weightsRew[rewardIndex] = 1;
    }

    /*!
     * Setter.
     * \param weights the weights to be set
     */
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
