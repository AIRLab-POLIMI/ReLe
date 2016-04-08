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

#include "rele/core/FiniteMDP.h"
#include "rele/utils/RandomGenerator.h"

using namespace std;

namespace ReLe
{

FiniteMDP::FiniteMDP(arma::cube P, arma::cube R, arma::cube Rsigma,
                     bool isFiniteHorizon, double gamma, unsigned int horizon) :
    Environment(), P(P), R(R), Rsigma(Rsigma)
{
    chekMatricesDimensions(P, R, Rsigma);
    setupenvironment(isFiniteHorizon, horizon, gamma, P);
    findAbsorbingStates();
}

FiniteMDP::FiniteMDP(arma::cube P, arma::cube R, arma::cube Rsigma, EnvironmentSettings *settings) :
    Environment(settings), P(P), R(R), Rsigma(Rsigma)
{
    chekMatricesDimensions(P, R, Rsigma);

    settings->statesNumber = P.n_cols;
    settings->actionsNumber = P.n_rows;
    settings->stateDimensionality = 1;
    settings->actionDimensionality = 1;

    findAbsorbingStates();
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
    currentState.setAbsorbing(absorbingStates.count(xn));
    nextState = currentState;

    //compute reward
    double m = R(u, x, xn);
    double sigma = Rsigma(u, x, xn);
    double r = RandomGenerator::sampleNormal(m, sigma);

    reward[0] = r;

}

void FiniteMDP::getInitialState(FiniteState& state)
{
    size_t x;
    do
    {
        x = RandomGenerator::sampleUniformInt(0, P.n_cols - 1);
    }
    while(absorbingStates.count(x) != 0);

    currentState.setStateN(x);
    state.setStateN(x);
}

void FiniteMDP::chekMatricesDimensions(const arma::cube& P, const arma::cube& R,
                                       const arma::cube& Rsigma)
{
    bool sameRows = (P.n_rows == R.n_rows) && (R.n_rows == Rsigma.n_rows);
    bool sameCols = (P.n_cols == R.n_cols) && (R.n_cols == Rsigma.n_cols);
    bool sameSlices = (P.n_slices == R.n_slices) && (R.n_slices == Rsigma.n_slices);
    bool sameState = P.n_cols == P.n_slices;

    if (!sameRows || !sameCols || !sameSlices || !sameState)
        throw invalid_argument("Invalid matrices:\n" //
                               "\t\tP must be [actions x states x states]\n"//
                               "\t\tR must be [actions x states x states]\n"//
                               "\t\tRsigma must be [actions x states x states]");
}

void FiniteMDP::setupenvironment(bool isFiniteHorizon, unsigned int horizon,
                                 double gamma, const arma::cube& P)
{
    EnvironmentSettings& task = getWritableSettings();
    task.isFiniteHorizon = isFiniteHorizon;
    task.horizon = horizon;
    task.gamma = gamma;
    task.isAverageReward = false;
    task.isEpisodic = false;
    task.statesNumber = P.n_cols;
    task.actionsNumber = P.n_rows;
    task.stateDimensionality = 1;
    task.actionDimensionality = 1;
    task.rewardDimensionality = 1;
}

void FiniteMDP::findAbsorbingStates()
{
    for(unsigned int state = 0; state < P.n_cols; state++)
    {
        unsigned int count = 0;
        for(unsigned int action = 0; action < P.n_rows; action++)
        {
            if(P(action, state, state) == 1.0)
                count++;
            else
                break;
        }

        if(count == P.n_rows)
            absorbingStates.insert(state);
    }
}

}
