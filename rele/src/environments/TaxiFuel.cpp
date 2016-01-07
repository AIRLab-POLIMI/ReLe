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

#include "rele/environments/TaxiFuel.h"
#include "rele/utils/RandomGenerator.h"

#include <iostream>

namespace ReLe
{

TaxiFuel::TaxiFuel() : DenseMDP(STATESIZE, ACTIONNUMBER, 1, false, true, 1.0), gridDim(0, 4)
{
    G = {4.0, 4.0};
    Y = {0.0, 0.0};
    B = {3.0, 0.0};
    R = {0.0, 4.0};
    F = {2.0, 1.0};

}


void TaxiFuel::step(const FiniteAction& action, DenseState& nextState,
                    Reward& reward)
{
    unsigned int u = action.getActionN();

    currentState[fuel] -= 1.0;
    reward[0] = -1;

    switch (u)
    {
    case up:
        currentState[y] = gridDim.bound(currentState[y] + 1);
        break;
    case down:
        currentState[y] = gridDim.bound(currentState[y] - 1);
        break;
    case left:
        currentState[x] = gridDim.bound(currentState[x] - 1);
        break;
    case right:
        currentState[x] = gridDim.bound(currentState[x] + 1);
        break;
    case pickup:
        if(currentState[onBoard] == 0 && atLocation())
            currentState[onBoard] = 1;
        else
            reward[0] -= 10;
        break;
    case dropoff:
        if(currentState[onBoard] == 1.0 && atDestination())
        {
            currentState[onBoard] = 2;
            currentState.setAbsorbing();
            reward[0] += 20;
        }
        else
        {
            reward[0] -= 10;
        }
        break;

    case fillup:
        if(atFuelStation())
            currentState[fuel] = 12;
        break;

    default:
        std::cerr << "Error occurred!" << std::endl;
        break;

    }

    if(currentState[fuel] < 0 && currentState[onBoard] != 2)
    {
        reward[0] -= 20;
        currentState.setAbsorbing();
    }

    nextState = currentState;
}

arma::vec2 TaxiFuel::extractTarget(int targetN)
{
    arma::vec2 target;
    switch (targetN)
    {
    case 0:
        target = G;
        break;
    case 1:
        target = Y;
        break;
    case 2:
        target = B;
        break;
    case 3:
        target = R;
        break;
    default:
        std::cerr << "Error occurred!" << std::endl;
        break;
    }
    return target;
}

bool TaxiFuel::atLocation()
{
    int targetN = currentState[location];
    arma::vec2 target = extractTarget(targetN);
    return target[x] == currentState[x] && target[y] == currentState[y];
}

bool TaxiFuel::atDestination()
{
    int targetN = currentState[destination];
    arma::vec2 target = extractTarget(targetN);
    return target[x] == currentState[x] && target[y] == currentState[y];
}

bool TaxiFuel::atFuelStation()
{
    return F[x] == currentState[x] && F[y] == currentState[y];
}

void TaxiFuel::getInitialState(DenseState& state)
{
    currentState[x] = RandomGenerator::sampleUniformInt(0, 4);
    currentState[y] = RandomGenerator::sampleUniformInt(0, 4);
    currentState[fuel] = RandomGenerator::sampleUniformInt(5, 12);
    currentState[onBoard] = 0;
    currentState[location] = RandomGenerator::sampleUniformInt(0, 3);

    int destinationStation;

    do
    {
        destinationStation = RandomGenerator::sampleUniformInt(0, 3);
    }
    while(currentState[location] == destinationStation);

    currentState[destination] = destinationStation;
    currentState.setAbsorbing(false);

    state = currentState;
}


}
