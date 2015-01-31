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

#include "SimpleChainGenerator.h"
using namespace std;

namespace ReLe
{

SimpleChainGenerator::SimpleChainGenerator()
{
    //setup algorithm data
    stateN = 0;
    actionN = 2;

    //default mdp parameters
    p = 0.8;
    rgoal = 1.0;
}

void SimpleChainGenerator::computeReward(size_t goalState)
{
    if (!isLeftmostState(goalState))
        R(RIGHT, previousState(goalState), goalState) = rgoal;

    if (!isRightmostState(goalState))
        R(LEFT, nextState(goalState), goalState) = rgoal;
}

void SimpleChainGenerator::computeprobabilities()
{
    for (size_t i = 0; i < stateN; i++)
    {
        if (isLeftmostState(i))
            P(LEFT, i, i) = 1.0;
        else
        {
            P(LEFT, i, i) = 1.0 - p;
            P(LEFT, i, previousState(i)) = p;
        }

        if (isRightmostState(i))
            P(RIGHT, i, i) = 1.0;
        else
        {
            P(RIGHT, i, i) = 1.0 - p;
            P(RIGHT, i, nextState(i)) = p;
        }
    }
}

void SimpleChainGenerator::generate(size_t size, size_t goalState)
{
    if (size <= 1)
        throw runtime_error("Chain must have at least two states");

    if (goalState >= size)
    {
        throw runtime_error(
            "goal state number must be lower than total states number");
    }

    stateN = size;
    P.zeros(actionN, stateN, stateN);
    R.zeros(actionN, stateN, stateN);
    Rsigma.zeros(actionN, stateN, stateN);

    computeReward(goalState);
    computeprobabilities();
}

}
