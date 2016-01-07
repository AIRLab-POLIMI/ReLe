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

#include "rele/environments/ShipSteering.h"

using namespace std;
using namespace arma;

namespace ReLe
{

ShipSteering::ShipSteering(bool small) :
    ContinuousMDP(STATESIZE, 1, 1, false, true, 0.99, 5000)
{
    double fieldSize = small ? 150 : 1000;

    rangeField = Range(0, fieldSize);
    rangeOmega = Range(-15, +15);

    gateS[0] = small ? 100 : 900;
    gateS[1] = small ? 120 : 920;

    gateE[0] = small ? 120 : 920;
    gateE[1] = small ? 100 : 900;
}

void ShipSteering::step(const DenseAction& action, DenseState& nextState,
                        Reward& reward)
{
    double r = rangeOmega.bound(action[0]);

    nextState[x] = currentState[x] + v*sin(currentState[theta])*dt;
    nextState[y] = currentState[y] + v*cos(currentState[theta])*dt;
    nextState[theta] = currentState[theta] + currentState[omega]*dt;
    nextState[omega] = currentState[omega] + (r - currentState[omega])*dt/T;

    if(!(rangeField.contains(nextState[x]) &&
            rangeField.contains(nextState[y])))
    {
        reward[0] = -10000;
        nextState.setAbsorbing();
    }
    else if(throughGate(currentState(span(x, y)), nextState(span(x, y))))
    {
        reward[0] = 0;
        nextState.setAbsorbing();
    }
    else
    {
        reward[0] = -1;
        nextState.setAbsorbing(false);
    }


    currentState = nextState;
}

void ShipSteering::getInitialState(DenseState& state)
{
    currentState[x] = 0;
    currentState[y] = 0;
    currentState[theta] = 0;
    currentState[omega] = 0;

    currentState.setAbsorbing(false);

    state = currentState;
}

bool ShipSteering::throughGate(const vec& start, const vec& end)
{
    vec r = gateE - gateS;
    vec s = end - start;
    double den = cross2D(r, s);

    if(den == 0)
        return false;


    double t = cross2D((start - gateS), s) / den;
    double u = cross2D((start - gateS), r) / den;

    return u >= 0 && u <= 1 && t >=0 && t <=1;
}

double ShipSteering::cross2D(const vec& v, const vec& w)
{
    return v[0]*w[1] - v[1]*w[0];
}

}
