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

#include "Rocky.h"
#include "Utils.h"

using namespace arma;
using namespace std;

namespace ReLe
{

Rocky::Rocky() :
    ContinuousMDP(STATESIZE, 3, 1, false, true), dt(0.2),
    maxOmega(M_PI), maxV(10)
{
    //TODO parameter in the constructor
    vec2 spot;
    spot[1] = 5;
    spot[2] = 0;
    foodSpots.push_back(spot);
}

void Rocky::step(const DenseAction& action, DenseState& nextState,
                 Reward& reward)
{
    double v = utils::threshold(action[0], maxV);
    double omega = utils::threshold(action[1], maxOmega);
    bool eat = (action[3] > 0 && v == 0 && omega == 0) ? true : false;

    //positions
    vec2 chickenPosition = currentState.rows(span(x, y));
    vec2 rockyRelPosition = currentState.rows(span(xr, yr));

    //update chicken position
    double thetaM = (2 * currentState[theta] + omega*dt) /2;
    currentState[x] += v * cos(thetaM) * dt;
    currentState[y] += v * sin(thetaM) * dt;
    currentState[theta] = utils::normalizeAngle(
                              currentState[theta] + omega * dt);

    //Compute sensors
    currentState[energy] = utils::threshold(currentState[energy] - 1, 0, 100);
    currentState[food] = 0;

    for (auto& spot : foodSpots)
    {
        if (norm(chickenPosition - spot) < 0.5)
        {
            currentState[food] = 1;

            if (eat)
            {
                currentState[energy] = utils::threshold(currentState[energy] + 5, 0, 100);
            }

            break;
        }
    }

    //Compute reward
    if (norm(rockyRelPosition) < 0.4)
    {
        reward[0] = -100;
        currentState.setAbsorbing(true);
    }
    else if (norm(chickenPosition) < 0.4 && currentState[energy] > 0)
    {
        reward[0] = currentState[energy];
        currentState.setAbsorbing(true);
    }
    else
    {
        reward[0] = 0;
    }
}

void Rocky::getInitialState(DenseState& state)
{
    //chicken state
    currentState[x] = 0;
    currentState[y] = 0;
    currentState[theta] = 0;

    //sensors state
    currentState[energy] = 0;
    currentState[food] = 0;

    //rocky state
    currentState[xr] = 5;
    currentState[yr] = 5;
    currentState[thetar] = 0;

    state = currentState;
}

}
