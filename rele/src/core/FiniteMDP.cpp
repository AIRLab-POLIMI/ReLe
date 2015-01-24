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

#include <stdexcept>

#include "FiniteMDP.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

FiniteMDP::FiniteMDP(arma::cube P, arma::cube R, bool isFiniteHorizon,
                     double gamma, unsigned int horizon) :
    Envirorment(), P(P), R(R)
{
    chekMatricesDimensions(P, R);
    setupEnvirorment(isFiniteHorizon, horizon, gamma, P);
}

void FiniteMDP::step(const FiniteAction& action, FiniteState& nextState,
                     Reward& reward)
{

    //Compute next state
    unsigned int u = action.getActionN();
    size_t x = currentState.getStateN();
    arma::vec prob = P.tube(u, x);
    size_t xn = RandomGenerator::sampleDiscrete(prob.begin(), prob.end());

    currentState.setStateN(xn);
    nextState.setStateN(xn);

    //compute reward
    double m = R(u, xn, 0);
    double sigma = R(u, xn, 1);
    double r = RandomGenerator::sampleNormal(m, sigma);

    reward.push_back(r);

}

void FiniteMDP::getInitialState(FiniteState& state)
{
    size_t x = RandomGenerator::sampleUniformInt(0, P.n_rows - 1);

    currentState.setStateN(x);
    state.setStateN(x);
}

void FiniteMDP::chekMatricesDimensions(const arma::cube& P, const arma::cube& R)
{
    if ((P.n_rows != R.n_rows) || (P.n_cols != R.n_cols)
            || (P.n_cols != P.n_slices) || (R.n_slices != 2))
        throw invalid_argument("Invalid matrices:\n" //
                               "\t\tP must be [actions x states x states]\n"//
                               "\t\tR must be [actions x states x 2]\n");
}

void FiniteMDP::setupEnvirorment(bool isFiniteHorizon, unsigned int horizon,
                                 double gamma, const arma::cube& P)
{
    EnvirormentSettings& task = getWritableSettings();
    task.isFiniteHorizon = isFiniteHorizon;
    task.horizon = horizon;
    task.gamma = gamma;
    task.isAverageReward = false;
    task.isEpisodic = false;
    task.finiteStateDim = P.n_cols;
    task.finiteActionDim = P.n_rows;
    task.continuosStateDim = 0;
    task.continuosActionDim = 0;
}

}
