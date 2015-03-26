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

#include "../../include/rele/environments/LQR.h"

#include "RandomGenerator.h"
#include <cassert>

using namespace std;

namespace ReLe
{

LQR::LQR(unsigned int dimension, unsigned int reward_dimension, double eps, double gamma, unsigned int horizon) :
    ContinuousMDP(dimension, dimension, reward_dimension, true, true, gamma, horizon),
    A(dimension,dimension), B(dimension,dimension)
{
    initialize(dimension, reward_dimension, eps);
}

LQR::LQR(arma::mat &A, arma::mat &B, std::vector<arma::mat> &Q, std::vector<arma::mat> &R, double gamma, unsigned int horizon) :
    ContinuousMDP(A.n_cols, B.n_cols, Q.size(), true, true, gamma, horizon),
    A(A), B(B), Q(Q), R(R)
{
    assert(Q.size() == R.size());
}

void LQR::step(const DenseAction& action,
               DenseState& nextState, Reward& reward)
{
    arma::vec& x = currentState;
    const arma::vec& u = action;
    for (unsigned int i = 0, ie = Q.size(); i < ie; ++i)
    {
        reward[i] = -((x.t()*Q[i]*x + u.t()*R[i]*u)[0]);
    }
    x = A*x + B*u;

    nextState = currentState;
    nextState.setAbsorbing(false);
}

void LQR::getInitialState(DenseState& state)
{
    for (unsigned int i = 0, ie = Q.size(); i < ie; ++i)
    {
        currentState[i] = -10;//RandomGenerator::sampleUniform(-20,20);
    }
    currentState.setAbsorbing(false);

    state = currentState;
}

void LQR::initialize(unsigned int stateActionSize, unsigned int rewardSize, double e)
{
    A.eye();
    B.eye();

    arma::mat IdMtx(stateActionSize, stateActionSize);
    IdMtx.eye();

    for (int i = 0, ie = rewardSize; i < ie; ++i)
    {
        Q.push_back(IdMtx);
        R.push_back(IdMtx);
    }

    if (stateActionSize > 1)
    {

        for (int i = 0, ie = stateActionSize; i < ie; ++i)
        {
            for (int j = 0, je = stateActionSize; j < je; ++j)
            {

                if (i == j)
                {
                    Q[i](j,j) = 1.0 - e;
                    R[i](j,j) = e;
                }
                else
                {
                    Q[i](j,j) = e;
                    R[i](j,j) = 1.0 - e;
                }
            }
        }
    }
}

}
