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

#include "rele/environments/MountainCar.h"
#include "rele/utils/RandomGenerator.h"

using namespace std;

namespace ReLe
{

MountainCar::MountainCar(ConfigurationsLabel label,
                         double initialPosition,
                         double initialVelocity) :
    // Sutton's article
    // DenseMDP(2, 3, 1, false, true),
    // Klein's articles
    // DenseMDP(2, 3, 1, false, true, 0.9, 100),
    // Ernst's article
    DenseMDP(2, 2, 1, false, true, 0.95, 100),
    envType(label),
    initialPosition(initialPosition),
    initialVelocity(initialVelocity)
{
}

void MountainCar::step(const FiniteAction& action,
                       DenseState& nextState, Reward& reward)
{
    if(envType == Sutton || envType == Klein)
    {
        int motorAction = action.getActionN() - 1;

        double updatedVelocity = currentState[velocity] + motorAction * 0.001
                                 - 0.0025 * cos(3 * currentState[position]);
        double updatedPosition = currentState[position] + updatedVelocity;


        if(updatedPosition <= -1.2)
        {
            currentState[position] = -1.2;
            currentState[velocity] = 0;
        }
        else if(updatedPosition > 0.5)
        {
            currentState[position] = 0.6;
            currentState[velocity] = 0;
            currentState.setAbsorbing();
        }
        else
        {
            currentState[position] = updatedPosition;
            currentState[velocity] = min(max(updatedVelocity, -0.07), 0.07);
        }

        if(envType == Sutton)
            reward[0] = -1;
        else if(envType == Klein)
        {
            if(currentState[position] > 0.5)
                reward[0] = 1;
            else
                reward[0] = 0;
        }
    }
    else if(envType == Ernst)
    {
        double diffHill;
        double diff2Hill;
        if(currentState[position] < 0)
        {
            diffHill = 2 * currentState[position] + 1;
            diff2Hill = 2;
        }
        else
        {
            diffHill = 1 / pow(1 + 5 * currentState[position] * currentState[position], 1.5);
            diff2Hill = (-15 * currentState[position]) /
                        pow(1 + 5 * currentState[position] * currentState[position], 2.5);
        }

        double h = 0.1;
        double m = 1;
        double g = 9.81;
        double u = -4 + (double(action.getActionN()) / (this->getSettings().actionsNumber - 1)) * 8;
        double acceleration = u / (m * (1 + diffHill * diffHill)) -
                              (g * diffHill) / (1 + diffHill * diffHill) -
                              (currentState[velocity] * currentState[velocity] * diffHill * diff2Hill) /
                              (1 + diffHill * diffHill);

        double updatedPosition = currentState[position] + h * currentState[velocity] +
                                 0.5 * h * h * acceleration;
        double updatedVelocity = currentState[velocity] + h * acceleration;

        if(abs(updatedPosition) > 1 || abs(updatedVelocity) > 3)
            currentState.setAbsorbing();

        currentState[position] = updatedPosition;
        currentState[velocity] = updatedVelocity;

        if(currentState[position] < -1 || abs(currentState[velocity]) > 3)
            reward[0] = -1;
        else if(currentState[position] > 1 && abs(currentState[velocity]) <= 3)
            reward[0] = 1;
        else
            reward[0] = 0;
    }

    nextState = currentState;
}

void MountainCar::getInitialState(DenseState& state)
{
    //Sutton's article
    if (envType == Sutton || envType == Ernst)
    {
        currentState[position] = initialPosition;
        currentState[velocity] =  initialVelocity;
    }
    else if(envType == Klein)
    {
        //Klein's article
        currentState[position] = RandomGenerator::sampleUniform(-1.2, -0.9);
        currentState[velocity] = RandomGenerator::sampleUniform(-0.07, 0.0);
    }
    else if(envType == Random)
    {
        currentState[position] = RandomGenerator::sampleUniform(-1.2,  0.5);
        currentState[velocity] = RandomGenerator::sampleUniform(-0.07, 0.07);
    }

    currentState.setAbsorbing(false);

    state = currentState;
}

}
