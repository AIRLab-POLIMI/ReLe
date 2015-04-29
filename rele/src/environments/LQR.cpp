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
    A(dimension,dimension), B(dimension,dimension), initialState(dimension)
{
    initialize(dimension, reward_dimension, eps);

    setInitialState();
}

LQR::LQR(arma::mat &A, arma::mat &B, std::vector<arma::mat> &Q, std::vector<arma::mat> &R, double gamma, unsigned int horizon) :
    ContinuousMDP(A.n_cols, B.n_cols, Q.size(), true, true, gamma, horizon),
    A(A), B(B), Q(Q), R(R)
{
    initialState.set_size(A.n_cols);
    assert(Q.size() == R.size());
    setInitialState();
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
    arma::vec& x = currentState;
    for (unsigned int i = 0; i < initialState.n_elem; ++i)
    {
        x(i) = RandomGenerator::sampleUniform(-3,3);//initialState;
    }
    currentState.setAbsorbing(false);

    state = currentState;
}

void LQR::setInitialState()
{
    for (unsigned int i = 0; i < initialState.n_elem; ++i)
    {
//        initialState[i] = RandomGenerator::sampleUniform(-3,3);//-10.0;
        initialState[i] = -10.0;
    }
}

void LQR::initialize(unsigned int dimensions, unsigned int rewardSize, double e)
{
    A.eye(dimensions, dimensions);
    B.eye(dimensions, dimensions);

    arma::mat IdMtx(dimensions, dimensions, arma::fill::eye);

    for (int i = 0; i < rewardSize; i++)
    {
        Q.push_back(IdMtx);
        R.push_back(IdMtx);
    }

    if (dimensions > 1)
    {

        for (int i = 0; i < rewardSize; i++)
        {
            for (int j = 0; j < dimensions; j++)
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
