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

#include "MountainCar.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

MountainCar::MountainCar() :
    //Sutton's article
//    DenseMDP(2, 3, 1, false, true)
    //Klein's articles
    DenseMDP(2, 3, 1, false, true, 0.9, 100)
{

}

void MountainCar::step(const FiniteAction& action,
                       DenseState& nextState, Reward& reward)
{
    int motorAction = action.getActionN() - 1;

    double computedVelocity = currentState[velocity] + motorAction * 0.001
                              - 0.0025 * cos(3 * currentState[position]);
    double computedPosition = currentState[position] + currentState[velocity];

    currentState[velocity] = min(max(computedVelocity, -0.07), 0.07);
    currentState[position] = min(max(computedPosition, -1.2), 0.6);

    //Sutton's article
//    if (currentState[position] <= -1.2)
//    {
//        currentState[velocity] = 0;
//    }
//    if (currentState[position] >= 0.6)
//    {
//        reward[0] = 0;
//        currentState.setAbsorbing();
//    }
//    else
//    {
//        reward[0] = -1;
//    }

    //Klein's article
    if (currentState[position] >= 0.5)
    {
        reward[0] = 1;
        currentState.setAbsorbing();
    }
    else
    {
        reward[0] = 0;
    }

    nextState = currentState;
}

void MountainCar::getInitialState(DenseState& state)
{
    //Sutton's article
//    currentState[position] = -0.5;
//    currentState[velocity] = 0.0;

    //Klein's article
    currentState[position] = RandomGenerator::sampleUniform(-1.2,-0.9);
    currentState[velocity] = RandomGenerator::sampleUniform(-0.07,0.0);

    currentState.setAbsorbing(false);

    state = currentState;
}

}
