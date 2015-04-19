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

#include "Solver.h"
#include "LQR.h"
#include "parametric/differentiable/LinearPolicy.h"

namespace ReLe
{

class LQRsolver: public Solver<DenseAction, DenseState>
{
public:
    LQRsolver(LQR& lqr, Features& approximator);
    virtual void solve();
    virtual Dataset<DenseAction, DenseState> test();
    virtual Policy<DenseAction, DenseState>& getPolicy();

    inline void setRewardIndex(unsigned int rewardIndex)
    {
        this->rewardIndex = rewardIndex;
    }

private:
    LQR& lqr;
    DetLinearPolicy<DenseState> pi;

    double gamma;

    unsigned int rewardIndex;
};

}

#endif /* INCLUDE_RELE_SOLVERS_LQRSOLVER_CPP_ */
