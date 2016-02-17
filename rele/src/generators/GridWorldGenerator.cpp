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

#include "rele/generators/GridWorldGenerator.h"
#include <iostream>

using namespace std;

namespace ReLe
{

GridWorldGenerator::GridWorldGenerator()
{
    //setup algorithm data
    stateN = 0;
    currentState = 0;
    actionN = 4;

    //default mdp parameters
    p = 0.89;
    rgoal = 10.0;
    rfall = -1.0;
    rstep = 0.0;
}

void GridWorldGenerator::load(const string& path)
{
    matrix.clear();
    stateNMatrix.clear();

    ifstream ifs;
    ifs.open(path);

    string line;
    while (getline(ifs, line))
    {
        cout << line << endl;
        stateN += count(line.begin(), line.end(), '0');
        stateN += count(line.begin(), line.end(), 'G');

        vector<char> chars(line.begin(), line.end());
        vector<int> numbers(line.length(), -1);
        matrix.push_back(chars);
        stateNMatrix.push_back(numbers);
    }

    P.zeros(actionN, stateN, stateN);
    R.zeros(actionN, stateN, stateN);
    Rsigma.zeros(actionN, stateN, stateN);
    currentState = 0;

    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[i].size(); j++)
        {
            assignStateNumbers(i, j);
        }
    }

    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[i].size(); j++)
        {
            handleChar(i, j);
        }
    }

}

void GridWorldGenerator::assignStateNumbers(size_t i, size_t j)
{
    switch (matrix[i][j])
    {
    case '0':
        stateNMatrix[i][j] = currentState++;
        break;

    case 'G':
        stateNMatrix[i][j] = stateN - 1;
        break;

    case '#':
        stateNMatrix[i][j] = -1;
        break;

    default:
        stateNMatrix[i][j] = -2;
        break;

    }

}

void GridWorldGenerator::handleChar(size_t i, size_t j)
{
    char c = matrix[i][j];

    if (c == '0' || c == 'G')
    {
        int currentS = stateNMatrix[i][j];

        for (int consideredS = 0; consideredS < stateN; consideredS++)
        {
            for (int action = 0; action < actionN; action++)
            {
                int actionS = getActionState(i, j, action);

                //compute probability to go from current to considered using action
                P(action, currentS, consideredS) = computeProbability(currentS,
                                                   consideredS, actionS);

                //compute mean reward
                R(action, currentS, consideredS) = computeReward(consideredS,
                                                   actionS);
            }
        }
    }

}

double GridWorldGenerator::computeProbability(int currentS, int consideredS,
        int actionS)
{
    if (currentS == getGoalStateN())
    {
        if (currentS == consideredS)
            return 1.0;
        else
            return 0;
    }

    if (actionS == consideredS)
        return p;

    if (currentS == consideredS && actionS >= 0)
        return 1.0 - p;

    if (currentS == consideredS && actionS < 0)
        return 1.0;

    return 0;

}

double GridWorldGenerator::computeReward(int consideredS, int actionS)
{
    if (actionS == -2)
        return rfall;

    if (actionS == consideredS && consideredS == getGoalStateN())
        return rgoal;

    return rstep;
}

int GridWorldGenerator::getGoalStateN()
{
    return stateN - 1;
}

int GridWorldGenerator::getActionState(std::size_t i, std::size_t j, int action)
{
    switch (action)
    {
    case N:
        i--;
        break;
    case S:
        i++;
        break;
    case W:
        j--;
        break;
    case E:
        j++;
        break;
    }

    if (i < stateNMatrix.size() && j < stateNMatrix[i].size())
    {
        return stateNMatrix[i][j];
    }
    else
    {
        return -2;
    }

}

}
