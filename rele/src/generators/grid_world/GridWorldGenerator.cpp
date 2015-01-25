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

#include "grid_world/GridWorldGenerator.h"

#include <iostream>

using namespace std;

namespace ReLe
{

GridWorldGenerator::GridWorldGenerator()
{
    stateN = 0;
    currentState = 0;
    actionN = 4;
}

void GridWorldGenerator::load(const string& path)
{
    string line;
    vector<vector<char>> matrix;
    vector<vector<int>> stateNMatrix;

    ifstream ifs;
    ifs.open(path);

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
    R.zeros(actionN, stateN, 2);
    currentState = 0;

    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[i].size(); j++)
        {
            assignStateNumbers(matrix, stateNMatrix, i, j);
        }
    }

    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[i].size(); j++)
        {
            handleChar(matrix, i, j);
        }
    }

}

FiniteMDP GridWorldGenerator::getMPD(double gamma)
{
    return FiniteMDP(P, R, false, gamma);
}

void GridWorldGenerator::assignStateNumbers(vector<vector<char>>& matrix,
        vector<vector<int>>& stateNMatrix, size_t i, size_t j)
{
    switch (matrix[i][j])
    {
    case '0':
        break;

    case 'G':
        break;

    default:
        break;

    }

}

void GridWorldGenerator::handleChar(std::vector<std::vector<char>>& matrix,
                                    std::size_t i, std::size_t j)
{
    char c = matrix[i][j];

    if (c == '0')
    {
        for (int action = 0; action < actionN; action++)
        {
            for (int state = 0; state < stateN; state++)
            {
                //P(action, currentState, )
            }
        }
    }

    currentState++;
}

}

